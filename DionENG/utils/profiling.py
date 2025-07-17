import time
import psutil
import numpy as np
import json
import os
from typing import Dict, List, Optional, Any
from collections import deque
from . import Logger, EventBus, Timer
from ..constants import PROFILER_LOG_INTERVAL, PROFILER_MAX_SAMPLES
from ..job_system import JobSystem, JobStatus, JobType
from ..math_lib import MathLib
from ..entity import Entity
from ..components import Transform, MeshRenderer, Physics3D, TacticalAI

class Profiler:
    """Profiling system for performance monitoring in DionENG."""
    def __init__(self, log_file: str = "profiler.log", max_samples: int = PROFILER_MAX_SAMPLES):
        self.logger = Logger(name="Profiler")
        self.event_bus = EventBus()
        self.timer = Timer()
        self.math = MathLib()
        self.max_samples = max_samples
        self.frame_times = deque(maxlen=max_samples)
        self.system_times: Dict[str, deque] = {}
        self.job_times: Dict[str, deque] = {}
        self.memory_stats: Dict[str, deque] = {
            "rss": deque(maxlen=max_samples),
            "vms": deque(maxlen=max_samples),
            "entity_count": deque(maxlen=max_samples),
            "component_count": deque(maxlen=max_samples)
        }
        self.job_system: Optional[JobSystem] = None
        self.log_file = log_file
        self.last_log_time = 0.0
        self.running = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize profiler and set up event listeners."""
        self.running = True
        self.logger.info(f"Initializing Profiler with max samples {self.max_samples}")
        self.event_bus.subscribe("job_completed", self._on_job_completed)
        self.event_bus.subscribe("job_failed", self._on_job_failed)
        self.event_bus.subscribe("system_update", self._on_system_update)
        self._setup_systems()

    def _setup_systems(self) -> None:
        """Initialize system timing deques."""
        systems = [
            "InputSystem", "AudioSystem", "PhysicsSystem2D", "PhysicsSystem3D",
            "RenderSystem2D", "RenderSystem3D", "AISystem", "NetworkSystem",
            "AssetManager", "MemorySystem", "VisibilitySystem", "ScriptSystem",
            "UISystem", "EditorSystem", "JobSystem"
        ]
        for system in systems:
            self.system_times[system] = deque(maxlen=self.max_samples)

    def set_job_system(self, job_system: JobSystem) -> None:
        """Set the job system for profiling job metrics."""
        self.job_system = job_system
        self.logger.info("JobSystem set for profiling")

    def start_frame(self) -> None:
        """Start profiling a new frame."""
        if not self.running:
            return
        self.frame_times.append(self.timer.delta_time)

    def profile_system(self, system_name: str, callback: Callable) -> Any:
        """Profile a system update."""
        if not self.running or system_name not in self.system_times:
            return callback()
        start_time = time.perf_counter()
        result = callback()
        elapsed = time.perf_counter() - start_time
        self.system_times[system_name].append(elapsed)
        self.event_bus.publish("system_update", {"system": system_name, "time": elapsed})
        return result

    def _on_system_update(self, data: Dict) -> None:
        """Handle system update event."""
        system = data.get("system")
        time_taken = data.get("time")
        if system in self.system_times:
            self.system_times[system].append(time_taken)

    def _on_job_completed(self, data: Dict) -> None:
        """Handle job completion event."""
        job_id = data.get("job_id")
        if self.job_system and job_id in self.job_system.job_map:
            job = self.job_system.job_map[job_id]
            self.job_times.setdefault(job.job_type.value, deque(maxlen=self.max_samples))
            self.job_times[job.job_type.value].append(job.end_time - job.start_time)

    def _on_job_failed(self, data: Dict) -> None:
        """Handle job failure event."""
        job_id = data.get("job_id")
        if self.job_system and job_id in self.job_system.job_map:
            job = self.job_system.job_map[job_id]
            self.job_times.setdefault(job.job_type.value, deque(maxlen=self.max_samples))
            self.job_times[job.job_type.value].append(job.end_time - job.start_time)

    def profile_memory(self, entities: List[Entity]) -> None:
        """Profile memory usage and entity/component counts."""
        if not self.running:
            return
        process = psutil.Process()
        mem_info = process.memory_info()
        self.memory_stats["rss"].append(mem_info.rss / 1024 / 1024)  # MB
        self.memory_stats["vms"].append(mem_info.vms / 1024 / 1024)  # MB
        self.memory_stats["entity_count"].append(len(entities))
        component_count = sum(len(entity.components) for entity in entities)
        self.memory_stats["component_count"].append(component_count)

    def profile_ray_tracing(self, rays: np.ndarray, scene: List[Entity]) -> Dict:
        """Profile a ray tracing job."""
        start_time = time.perf_counter()
        results = []
        for ray_origin, ray_dir in rays:
            for entity in scene:
                mesh = entity.get_component("MeshRenderer")
                transform = entity.get_component("Transform")
                if mesh and transform:
                    for i in range(0, len(mesh.indices), 3):
                        v0 = transform.world_matrix @ mesh.vertices[mesh.indices[i]]
                        v1 = transform.world_matrix @ mesh.vertices[mesh.indices[i + 1]]
                        v2 = transform.world_matrix @ mesh.vertices[mesh.indices[i + 2]]
                        hit = self.math.ray_triangle_intersection(ray_origin, ray_dir, v0, v1, v2)
                        if hit:
                            results.append({"entity_id": entity.id, "hit_point": hit})
        elapsed = time.perf_counter() - start_time
        self.job_times.setdefault("ray_tracing", deque(maxlen=self.max_samples))
        self.job_times["ray_tracing"].append(elapsed)
        return {"rays_processed": len(rays), "hits": len(results), "time": elapsed}

    def profile_opencl_job(self, kernel_name: str, data: np.ndarray) -> Dict:
        """Profile an OpenCL compute job."""
        start_time = time.perf_counter()
        # Placeholder for OpenCL execution
        result = data * 2.0
        elapsed = time.perf_counter() - start_time
        self.job_times.setdefault("opencl", deque(maxlen=self.max_samples))
        self.job_times["opencl"].append(elapsed)
        return {"kernel": kernel_name, "data_size": data.size, "time": elapsed}

    def profile_quadtree_query(self, entities: List[Entity], bounds: Tuple) -> Dict:
        """Profile a quadtree query."""
        start_time = time.perf_counter()
        results = []
        for entity in entities:
            transform = entity.get_component("Transform")
            if transform and self.math.point_in_bounds(transform.position[:2], bounds):
                results.append(entity.id)
        elapsed = time.perf_counter() - start_time
        self.job_times.setdefault("quadtree", deque(maxlen=self.max_samples))
        self.job_times["quadtree"].append(elapsed)
        return {"entities_queried": len(entities), "matches": len(results), "time": elapsed}

    def profile_dynamic_resolution(self, viewport: Tuple[int, int], scale_factor: float) -> Dict:
        """Profile dynamic resolution scaling."""
        start_time = time.perf_counter()
        new_width = int(viewport[0] * scale_factor)
        new_height = int(viewport[1] * scale_factor)
        elapsed = time.perf_counter() - start_time
        self.job_times.setdefault("dynamic_resolution", deque(maxlen=self.max_samples))
        self.job_times["dynamic_resolution"].append(elapsed)
        return {"new_width": new_width, "new_height": new_height, "time": elapsed}

    def profile_physics(self, entities: List[Entity], dt: float) -> Dict:
        """Profile physics calculations."""
        start_time = time.perf_counter()
        results = []
        for entity in entities:
            physics = entity.get_component("Physics3D")
            transform = entity.get_component("Transform")
            if physics and transform:
                physics.apply_force(self.math.vec3(0, -9.81 * physics.mass, 0))  # Gravity
                transform.set_position(self.math.vec_add(
                    transform.position,
                    self.math.vec_scale(physics.velocity, dt)
                ))
                results.append({"entity_id": entity.id, "new_position": transform.position})
        elapsed = time.perf_counter() - start_time
        self.system_times.setdefault("PhysicsSystem3D", deque(maxlen=self.max_samples))
        self.system_times["PhysicsSystem3D"].append(elapsed)
        return {"entities_processed": len(results), "time": elapsed}

    def profile_ai(self, entity: Entity, grid: np.ndarray, target: np.ndarray) -> Dict:
        """Profile AI pathfinding."""
        start_time = time.perf_counter()
        ai = entity.get_component("TacticalAI")
        transform = entity.get_component("Transform")
        result = None
        if ai and transform:
            result = ai.update_path(transform.position, target, grid)
        elapsed = time.perf_counter() - start_time
        self.system_times.setdefault("AISystem", deque(maxlen=self.max_samples))
        self.system_times["AISystem"].append(elapsed)
        return {"entity_id": entity.id, "path_length": len(result) if result else 0, "time": elapsed}

    def get_metrics(self) -> Dict:
        """Get comprehensive profiling metrics."""
        frame_times = list(self.frame_times)
        avg_frame_time = np.mean(frame_times) if frame_times else 0.0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        metrics = {
            "fps": fps,
            "avg_frame_time_ms": avg_frame_time * 1000,
            "frame_time_samples": len(frame_times),
            "system_times": {
                system: {
                    "avg_ms": np.mean(list(times)) * 1000 if times else 0.0,
                    "max_ms": np.max(list(times)) * 1000 if times else 0.0,
                    "min_ms": np.min(list(times)) * 1000 if times else 0.0,
                    "samples": len(times)
                } for system, times in self.system_times.items()
            },
            "job_times": {
                job_type: {
                    "avg_ms": np.mean(list(times)) * 1000 if times else 0.0,
                    "max_ms": np.max(list(times)) * 1000 if times else 0.0,
                    "min_ms": np.min(list(times)) * 1000 if times else 0.0,
                    "samples": len(times)
                } for job_type, times in self.job_times.items()
            },
            "memory": {
                "rss_mb": np.mean(list(self.memory_stats["rss"])) if self.memory_stats["rss"] else 0.0,
                "vms_mb": np.mean(list(self.memory_stats["vms"])) if self.memory_stats["vms"] else 0.0,
                "entity_count": np.mean(list(self.memory_stats["entity_count"])) if self.memory_stats["entity_count"] else 0.0,
                "component_count": np.mean(list(self.memory_stats["component_count"])) if self.memory_stats["component_count"] else 0.0
            }
        }
        return metrics

    def log_metrics(self) -> None:
        """Log metrics to console and file if interval reached."""
        if not self.running or self.timer.get_elapsed() - self.last_log_time < PROFILER_LOG_INTERVAL:
            return
        metrics = self.get_metrics()
        log_message = (
            f"Profiler Metrics:\n"
            f"FPS: {metrics['fps']:.2f}, Avg Frame Time: {metrics['avg_frame_time_ms']:.2f}ms\n"
            f"Systems:\n" +
            "\n".join(
                f"  {system}: Avg {data['avg_ms']:.2f}ms, Max {data['max_ms']:.2f}ms, Min {data['min_ms']:.2f}ms, Samples {data['samples']}"
                for system, data in metrics["system_times"].items() if data["samples"] > 0
            ) +
            "\nJobs:\n" +
            "\n".join(
                f"  {job_type}: Avg {data['avg_ms']:.2f}ms, Max {data['max_ms']:.2f}ms, Min {data['min_ms']:.2f}ms, Samples {data['samples']}"
                for job_type, data in metrics["job_times"].items() if data["samples"] > 0
            ) +
            f"\nMemory: RSS {metrics['memory']['rss_mb']:.2f}MB, VMS {metrics['memory']['vms_mb']:.2f}MB, "
            f"Entities {metrics['memory']['entity_count']:.0f}, Components {metrics['memory']['component_count']:.0f}"
        )
        self.logger.info(log_message)
        self._save_metrics_to_file(metrics)
        self.last_log_time = self.timer.get_elapsed()

    def _save_metrics_to_file(self, metrics: Dict) -> None:
        """Save metrics to a JSON file."""
        try:
            with open(self.log_file, 'a') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics
                }, f, indent=2)
                f.write("\n")
            self.logger.debug(f"Saved metrics to {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {self.log_file}: {e}")

    def generate_report(self, output_file: str) -> bool:
        """Generate a detailed performance report."""
        try:
            metrics = self.get_metrics()
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "summary": {
                    "bottlenecks": self._identify_bottlenecks(),
                    "recommendations": self._generate_recommendations()
                }
            }
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Generated report to {output_file}")
            self.event_bus.publish("report_generated", {"file": output_file})
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return False

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        metrics = self.get_metrics()
        for system, data in metrics["system_times"].items():
            if data["avg_ms"] > 10.0:  # Threshold for slow systems
                bottlenecks.append(f"System {system}: High avg time {data['avg_ms']:.2f}ms")
        for job_type, data in metrics["job_times"].items():
            if data["avg_ms"] > 5.0:  # Threshold for slow jobs
                bottlenecks.append(f"Job {job_type}: High avg time {data['avg_ms']:.2f}ms")
        if metrics["memory"]["rss_mb"] > 1000:  # Threshold for high memory usage
            bottlenecks.append(f"High memory usage: RSS {metrics['memory']['rss_mb']:.2f}MB")
        return bottlenecks

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        metrics = self.get_metrics()
        if metrics["fps"] < 30:
            recommendations.append("Optimize system updates to improve FPS (current: {:.2f})".format(metrics["fps"]))
        for system, data in metrics["system_times"].items():
            if data["avg_ms"] > 10.0:
                recommendations.append(f"Optimize {system}: Reduce avg update time ({data['avg_ms']:.2f}ms)")
        for job_type, data in metrics["job_times"].items():
            if data["avg_ms"] > 5.0:
                recommendations.append(f"Optimize {job_type} jobs: Reduce avg execution time ({data['avg_ms']:.2f}ms)")
        if metrics["memory"]["rss_mb"] > 1000:
            recommendations.append("Reduce memory usage: Current RSS {:.2f}MB".format(metrics["memory"]["rss_mb"]))
        return recommendations

    def shutdown(self) -> None:
        """Shutdown the profiler."""
        self.running = False
        self.log_metrics()
        self.generate_report("profiler_final_report.json")
        self.logger.info("Profiler shutdown complete")
        self.frame_times.clear()
        self.system_times.clear()
        self.job_times.clear()
        self.memory_stats.clear()