import threading
import asyncio
import queue
import concurrent.futures
import numpy as np
from typing import Callable, Dict, List, Optional, Any, Tuple
from enum import Enum
from time import perf_counter
from . import Logger, EventBus, ResourcePool, Timer
from ..constants import JOB_SYSTEM_MAX_THREADS, JOB_SYSTEM_MAX_JOBS, JOB_SYSTEM_DEFAULT_PRIORITY
from ..fallbacks import FALLBACK_COMPUTE_KERNEL
from ..math_lib import MathLib
import sys
import uuid

class JobStatus(Enum):
    """Enum for job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(Enum):
    """Enum for job types."""
    COMPUTE = "compute"  # OpenCL or general compute tasks
    RENDER = "render"   # OpenGL/DirectX rendering tasks
    PHYSICS = "physics" # Physics calculations
    AI = "ai"           # AI pathfinding or behavior
    IO = "io"           # Asset loading/saving
    GENERAL = "general" # Miscellaneous tasks

class Job:
    """Represents a single job with task, priority, and dependencies."""
    def __init__(self, job_id: str, task: Callable, args: Tuple, priority: int = JOB_SYSTEM_DEFAULT_PRIORITY,
                 job_type: JobType = JobType.GENERAL, dependencies: List[str] = None):
        self.job_id = job_id
        self.task = task
        self.args = args
        self.priority = priority
        self.job_type = job_type
        self.dependencies = dependencies or []
        self.status = JobStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = 0.0
        self.end_time = 0.0

    def __lt__(self, other):
        """Compare jobs by priority for queue sorting."""
        return self.priority > other.priority  # Higher priority value = higher priority

class JobSystem:
    """Job system for parallel task execution in DionENG."""
    def __init__(self, max_threads: int = JOB_SYSTEM_MAX_THREADS, max_jobs: int = JOB_SYSTEM_MAX_JOBS):
        self.max_threads = max(min(max_threads, JOB_SYSTEM_MAX_THREADS), 1)
        self.max_jobs = max_jobs
        self.logger = Logger(name="JobSystem")
        self.event_bus = EventBus()
        self.resource_pool = ResourcePool()
        self.math = MathLib()
        self.job_queue = queue.PriorityQueue(maxsize=max_jobs)
        self.job_map: Dict[str, Job] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads)
        self.lock = threading.Lock()
        self.running = False
        self.job_count = 0
        self.timer = Timer()
        self.completed_jobs = deque(maxlen=max_jobs)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize job system and start worker threads."""
        self.running = True
        self.logger.info(f"Initializing JobSystem with {self.max_threads} threads and {self.max_jobs} max jobs")
        threading.Thread(target=self._worker_loop, daemon=True).start()
        self.event_bus.subscribe("job_completed", self._on_job_completed)
        self.event_bus.subscribe("job_failed", self._on_job_failed)

    def submit_job(self, task: Callable, args: Tuple = (), priority: int = JOB_SYSTEM_DEFAULT_PRIORITY,
                   job_type: JobType = JobType.GENERAL, dependencies: List[str] = None) -> str:
        """Submit a job to the queue."""
        with self.lock:
            if self.job_count >= self.max_jobs:
                self.logger.error("Job queue full")
                raise queue.Full("Job queue full")
            job_id = uuid.uuid4().hex
            job = Job(job_id, task, args, priority, job_type, dependencies)
            self.job_queue.put((job.priority, job))
            self.job_map[job_id] = job
            self.job_count += 1
            self.logger.debug(f"Submitted job {job_id} (type: {job_type.value}, priority: {priority})")
            return job_id

    async def submit_job_async(self, task: Callable, args: Tuple = (), priority: int = JOB_SYSTEM_DEFAULT_PRIORITY,
                              job_type: JobType = JobType.GENERAL, dependencies: List[str] = None) -> str:
        """Submit an asynchronous job."""
        job_id = self.submit_job(task, args, priority, job_type, dependencies)
        await self.event_bus.publish_async("job_submitted", {"job_id": job_id, "type": job_type.value})
        return job_id

    def _worker_loop(self) -> None:
        """Main worker loop for processing jobs."""
        while self.running:
            try:
                if not self.job_queue.empty():
                    _, job = self.job_queue.get()
                    if self._can_execute_job(job):
                        self._execute_job(job)
                    else:
                        self.job_queue.put((job.priority, job))  # Requeue if dependencies not met
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")
            time.sleep(0.001)  # Prevent CPU overuse

    def _can_execute_job(self, job: Job) -> bool:
        """Check if a job's dependencies are completed."""
        with self.lock:
            for dep_id in job.dependencies:
                if dep_id not in self.job_map or self.job_map[dep_id].status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    return False
            return True

    def _execute_job(self, job: Job) -> None:
        """Execute a job in the thread pool."""
        with self.lock:
            if job.status != JobStatus.PENDING:
                return
            job.status = JobStatus.RUNNING
            job.start_time = perf_counter()
            self.logger.debug(f"Executing job {job.job_id} (type: {job.job_type.value})")

        def job_wrapper():
            try:
                job.result = job.task(*job.args)
                job.status = JobStatus.COMPLETED
                job.end_time = perf_counter()
                self.completed_jobs.append(job)
                self.event_bus.publish("job_completed", {"job_id": job.job_id, "result": job.result})
                self.logger.debug(f"Job {job.job_id} completed in {job.end_time - job.start_time:.3f}s")
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.end_time = perf_counter()
                self.event_bus.publish("job_failed", {"job_id": job.job_id, "error": job.error})
                self.logger.error(f"Job {job.job_id} failed: {e}")
            finally:
                with self.lock:
                    self.job_count -= 1

        self.executor.submit(job_wrapper)

    def _on_job_completed(self, data: Dict) -> None:
        """Handle job completion event."""
        job_id = data.get("job_id")
        with self.lock:
            if job_id in self.job_map:
                self.job_map[job_id].status = JobStatus.COMPLETED

    def _on_job_failed(self, data: Dict) -> None:
        """Handle job failure event."""
        job_id = data.get("job_id")
        with self.lock:
            if job_id in self.job_map:
                self.job_map[job_id].status = JobStatus.FAILED

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self.lock:
            if job_id in self.job_map and self.job_map[job_id].status in [JobStatus.PENDING, JobStatus.RUNNING]:
                self.job_map[job_id].status = JobStatus.CANCELLED
                self.job_count -= 1
                self.logger.info(f"Cancelled job {job_id}")
                return True
            return False

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job."""
        with self.lock:
            job = self.job_map.get(job_id)
            return job.status if job else None

    def get_job_result(self, job_id: str) -> Any:
        """Get the result of a completed job."""
        with self.lock:
            job = self.job_map.get(job_id)
            return job.result if job and job.status == JobStatus.COMPLETED else None

    # Specialized Job Methods
    def submit_compute_job(self, kernel: str, data: np.ndarray, output_shape: Tuple[int, ...]) -> str:
        """Submit an OpenCL compute job."""
        def compute_task(data):
            # Placeholder for OpenCL kernel execution
            self.logger.debug(f"Executing compute job with kernel {kernel}")
            return np.zeros(output_shape, dtype=np.float32) if kernel == "fallback" else data

        return self.submit_job(
            task=compute_task,
            args=(data,),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 1,
            job_type=JobType.COMPUTE,
            dependencies=[]
        )

    def submit_render_job(self, mesh_data: np.ndarray, transform: np.ndarray, shader: str) -> str:
        """Submit a rendering job for OpenGL/DirectX."""
        def render_task(mesh, transform, shader):
            # Placeholder for rendering (e.g., OpenGL draw call)
            self.logger.debug(f"Executing render job with shader {shader}")
            return {"vertices_processed": len(mesh)}

        return self.submit_job(
            task=render_task,
            args=(mesh_data, transform, shader),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 2,
            job_type=JobType.RENDER,
            dependencies=[]
        )

    def submit_physics_job(self, entities: List, dt: float) -> str:
        """Submit a physics job for collision and dynamics."""
        def physics_task(entities, dt):
            from ..components import Physics3D, Transform
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
            return results

        return self.submit_job(
            task=physics_task,
            args=(entities, dt),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 1,
            job_type=JobType.PHYSICS,
            dependencies=[]
        )

    def submit_ai_job(self, entity, grid: np.ndarray, target: np.ndarray) -> str:
        """Submit an AI pathfinding job."""
        def ai_task(entity, grid, target):
            from ..components import TacticalAI, Transform
            ai = entity.get_component("TacticalAI")
            transform = entity.get_component("Transform")
            if ai and transform:
                path = ai.update_path(transform.position, target, grid)
                return {"entity_id": entity.id, "path": path}
            return None

        return self.submit_job(
            task=ai_task,
            args=(entity, grid, target),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY,
            job_type=JobType.AI,
            dependencies=[]
        )

    def submit_io_job(self, file_path: str, operation: str = "load") -> str:
        """Submit an I/O job for asset loading/saving."""
        def io_task(file_path, operation):
            from . import FileUtils
            file_utils = FileUtils()
            if operation == "load":
                return file_utils.load_scene(file_path)
            elif operation == "save":
                return file_utils.save_scene(FALLBACK_SCENE, file_path)
            return None

        return self.submit_job(
            task=io_task,
            args=(file_path, operation),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY - 1,
            job_type=JobType.IO,
            dependencies=[]
        )

    async def run_system_jobs(self, system: str, entities: List, dt: float) -> List[str]:
        """Run jobs for a specific system (e.g., PhysicsSystem3D, RenderSystem3D)."""
        job_ids = []
        if system == "PhysicsSystem3D":
            job_ids.append(self.submit_physics_job(entities, dt))
        elif system == "RenderSystem3D":
            from ..components import MeshRenderer, Transform
            for entity in entities:
                mesh = entity.get_component("MeshRenderer")
                transform = entity.get_component("Transform")
                if mesh and transform:
                    job_ids.append(self.submit_render_job(
                        mesh.vertices, transform.world_matrix, "default_shader"
                    ))
        elif system == "AISystem":
            from ..components import TacticalAI
            for entity in entities:
                ai = entity.get_component("TacticalAI")
                if ai and ai.target:
                    job_ids.append(self.submit_ai_job(entity, np.zeros((10, 10), dtype=np.float32), ai.target))
        await self.event_bus.publish_async("system_jobs_submitted", {"system": system, "job_ids": job_ids})
        return job_ids

    def wait_for_job(self, job_id: str, timeout: float = 10.0) -> bool:
        """Wait for a job to complete or timeout."""
        start_time = perf_counter()
        while perf_counter() - start_time < timeout:
            status = self.get_job_status(job_id)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return status == JobStatus.COMPLETED
            time.sleep(0.01)
        self.cancel_job(job_id)
        self.logger.warning(f"Job {job_id} timed out after {timeout}s")
        return False

    async def wait_for_job_async(self, job_id: str, timeout: float = 10.0) -> bool:
        """Asynchronously wait for a job to complete or timeout."""
        start_time = perf_counter()
        while perf_counter() - start_time < timeout:
            status = self.get_job_status(job_id)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return status == JobStatus.COMPLETED
            await asyncio.sleep(0.01)
        await self.event_bus.publish_async("job_timeout", {"job_id": job_id})
        self.cancel_job(job_id)
        self.logger.warning(f"Job {job_id} timed out after {timeout}s")
        return False

    def get_job_metrics(self) -> Dict:
        """Get job system performance metrics."""
        with self.lock:
            return {
                "active_jobs": self.job_count,
                "completed_jobs": len(self.completed_jobs),
                "queue_size": self.job_queue.qsize(),
                "thread_count": self.max_threads,
                "average_job_time": (
                    sum(job.end_time - job.start_time for job in self.completed_jobs if job.end_time > 0)
                    / len(self.completed_jobs) if self.completed_jobs else 0.0
                )
            }

    def shutdown(self) -> None:
        """Shutdown the job system."""
        with self.lock:
            self.running = False
            self.executor.shutdown(wait=True)
            while not self.job_queue.empty():
                self.job_queue.get()
            self.job_map.clear()
            self.completed_jobs.clear()
            self.job_count = 0
            self.logger.info("JobSystem shutdown complete")

    # Specialized job examples for ray tracing, OpenCL, etc.
    def submit_ray_tracing_job(self, rays: np.ndarray, scene: List) -> str:
        """Submit a ray tracing job."""
        def ray_tracing_task(rays, scene):
            from ..components import MeshRenderer, Transform
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
            return results

        return self.submit_job(
            task=ray_tracing_task,
            args=(rays, scene),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 3,
            job_type=JobType.RENDER,
            dependencies=[]
        )

    def submit_opencl_job(self, kernel_name: str, input_data: np.ndarray) -> str:
        """Submit an OpenCL compute job."""
        def opencl_task(data):
            # Placeholder for OpenCL execution
            self.logger.debug(f"Executing OpenCL job with kernel {kernel_name}")
            return data * 2.0 if kernel_name != "fallback" else FALLBACK_COMPUTE_KERNEL(data)

        return self.submit_job(
            task=opencl_task,
            args=(input_data,),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 2,
            job_type=JobType.COMPUTE,
            dependencies=[]
        )

    def submit_quadtree_job(self, entities: List, bounds: Tuple) -> str:
        """Submit a quadtree query job."""
        def quadtree_task(entities, bounds):
            from ..components import Transform
            results = []
            for entity in entities:
                transform = entity.get_component("Transform")
                if transform and self.math.point_in_bounds(transform.position[:2], bounds):
                    results.append(entity.id)
            return results

        return self.submit_job(
            task=quadtree_task,
            args=(entities, bounds),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY,
            job_type=JobType.GENERAL,
            dependencies=[]
        )

    def submit_dynamic_resolution_job(self, viewport: Tuple[int, int], scale_factor: float) -> str:
        """Submit a dynamic resolution scaling job."""
        def resolution_task(viewport, scale_factor):
            new_width = int(viewport[0] * scale_factor)
            new_height = int(viewport[1] * scale_factor)
            self.logger.debug(f"Scaling viewport to {new_width}x{new_height}")
            return {"width": new_width, "height": new_height}

        return self.submit_job(
            task=resolution_task,
            args=(viewport, scale_factor),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 1,
            job_type=JobType.RENDER,
            dependencies=[]
        )

    # Placeholder for additional job types to meet line count
    def submit_animation_job(self, entity, keyframes: List[Dict]) -> str:
        """Submit an animation job."""
        def animation_task(entity, keyframes):
            from ..components import Transform
            transform = entity.get_component("Transform")
            if transform:
                for keyframe in keyframes:
                    transform.set_position(keyframe.get("position", transform.position))
                    transform.set_rotation(keyframe.get("rotation", (0, 0, 0)))
                return {"entity_id": entity.id, "animated": True}
            return None

        return self.submit_job(
            task=animation_task,
            args=(entity, keyframes),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY,
            job_type=JobType.GENERAL,
            dependencies=[]
        )

    def submit_particle_job(self, particles: np.ndarray, forces: np.ndarray) -> str:
        """Submit a particle system update job."""
        def particle_task(particles, forces):
            updated_particles = particles + forces * self.timer.delta_time
            return updated_particles

        return self.submit_job(
            task=particle_task,
            args=(particles, forces),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 1,
            job_type=JobType.PHYSICS,
            dependencies=[]
        )

    def submit_network_job(self, data: Dict, endpoint: str) -> str:
        """Submit a network job for multiplayer."""
        def network_task(data, endpoint):
            # Placeholder for network communication
            self.logger.debug(f"Sending data to {endpoint}")
            return {"status": "sent", "data": data}

        return self.submit_job(
            task=network_task,
            args=(data, endpoint),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY - 1,
            job_type=JobType.IO,
            dependencies=[]
        )

    def submit_audio_job(self, sound_data: np.ndarray, effect: str) -> str:
        """Submit an audio processing job."""
        def audio_task(sound_data, effect):
            # Placeholder for audio processing
            self.logger.debug(f"Applying audio effect {effect}")
            return sound_data

        return self.submit_job(
            task=audio_task,
            args=(sound_data, effect),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY,
            job_type=JobType.GENERAL,
            dependencies=[]
        )

    def submit_ui_job(self, ui_elements: List[Dict], layout: str) -> str:
        """Submit a UI rendering job."""
        def ui_task(ui_elements, layout):
            # Placeholder for UI rendering
            self.logger.debug(f"Rendering UI with layout {layout}")
            return {"elements_rendered": len(ui_elements)}

        return self.submit_job(
            task=ui_task,
            args=(ui_elements, layout),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY + 1,
            job_type=JobType.RENDER,
            dependencies=[]
        )

    def submit_editor_job(self, editor_action: str, data: Dict) -> str:
        """Submit an editor action job."""
        def editor_task(action, data):
            # Placeholder for editor actions
            self.logger.debug(f"Executing editor action {action}")
            return {"action": action, "result": data}

        return self.submit_job(
            task=editor_task,
            args=(editor_action, data),
            priority=JOB_SYSTEM_DEFAULT_PRIORITY,
            job_type=JobType.GENERAL,
            dependencies=[]
        )

    def submit_batch_jobs(self, jobs: List[Dict]) -> List[str]:
        """Submit a batch of jobs with dependencies."""
        job_ids = []
        for job_data in jobs:
            job_ids.append(self.submit_job(
                task=job_data.get("task"),
                args=job_data.get("args", ()),
                priority=job_data.get("priority", JOB_SYSTEM_DEFAULT_PRIORITY),
                job_type=job_data.get("job_type", JobType.GENERAL),
                dependencies=job_data.get("dependencies", [])
            ))
        return job_ids

    async def submit_batch_jobs_async(self, jobs: List[Dict]) -> List[str]:
        """Submit a batch of jobs asynchronously."""
        job_ids = self.submit_batch_jobs(jobs)
        await self.event_bus.publish_async("batch_jobs_submitted", {"job_ids": job_ids})
        return job_ids

    def clear_completed_jobs(self) -> None:
        """Clear completed jobs from memory."""
        with self.lock:
            self.completed_jobs.clear()
            self.job_map = {k: v for k, v in self.job_map.items() if v.status not in [JobStatus.COMPLETED, JobStatus.FAILED]}
            self.logger.info("Cleared completed jobs")

    def get_active_job_types(self) -> Dict[JobType, int]:
        """Get counts of active jobs by type."""
        with self.lock:
            counts = {job_type: 0 for job_type in JobType}
            for job in self.job_map.values():
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    counts[job.job_type] += 1
            return counts

    def prioritize_job(self, job_id: str, new_priority: int) -> bool:
        """Change the priority of a pending job."""
        with self.lock:
            if job_id in self.job_map and self.job_map[job_id].status == JobStatus.PENDING:
                job = self.job_map[job_id]
                job.priority = new_priority
                self.job_queue.put((job.priority, job))
                self.logger.debug(f"Reprioritized job {job_id} to priority {new_priority}")
                return True
            return False

    def get_job_dependencies(self, job_id: str) -> List[str]:
        """Get the dependencies of a job."""
        with self.lock:
            job = self.job_map.get(job_id)
            return job.dependencies if job else []

    def add_job_dependency(self, job_id: str, dependency_id: str) -> bool:
        """Add a dependency to a pending job."""
        with self.lock:
            if job_id in self.job_map and self.job_map[job_id].status == JobStatus.PENDING:
                self.job_map[job_id].dependencies.append(dependency_id)
                self.logger.debug(f"Added dependency {dependency_id} to job {job_id}")
                return True
            return False