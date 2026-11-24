import { vec3 } from "wgpu-matrix";
import CameraController from "./CameraController";
import Renderer from "./Renderer";
import Scene, { type SceneOptions } from "./Scene";
import Simulation from "./Simulation";
import WebGpuContext from "./WebGpuContext";
import "./style.css"
import { createControlPanel } from "./ControlPanel";

const gpuProfiling = new URLSearchParams(window.location.search).has("gpuProfiling");
const gpuProfilingWarmupSteps = 4800;

const canvas = document.querySelector<HTMLCanvasElement>("#gpu-canvas");

if (!canvas) {
  throw new Error("Canvas #gpu-canvas was not found");
}

const gpuContext = await WebGpuContext.create(canvas);
const { device } = gpuContext;

const sceneOptions: SceneOptions = {
  seed: 0,
  initialBodyCount: 2000,
  initialVelocityRange: 10,
  simulationConfig: {
    capacity: 100000,
    fixedDt: 1 / 120,
    gravitationalConstant: 0.05,
    softeningLength: 0.1,
    linearDragRate: 1.0,
    starMass: 80,
    explosionMass: 160
  },
  boundary: {
    enabled: true,
    min: vec3.create(-50, -50, -50), 
    max: vec3.create(50, 50, 50)
  }
};

const scene = new Scene(sceneOptions);

const simulation = new Simulation(device, scene);

const renderer = new Renderer(
  gpuContext, 
  simulation.bodiesBuffer, 
  simulation.collisionCandidatesBuffer,
  simulation.simulationParamsBuffer,
  simulation.simulationDispatchArgsBuffer
);

if (gpuProfiling) {
  const stepsPerBatch = 16;
  for (let batchStart = 0; batchStart < gpuProfilingWarmupSteps; batchStart += stepsPerBatch) {
    const encoder = device.createCommandEncoder();
    const stepsInBatch = Math.min(stepsPerBatch, gpuProfilingWarmupSteps - batchStart);
    for (let step = 0; step < stepsInBatch; step++) {
      simulation.encodeStep(encoder);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
}

const cameraController = new CameraController(canvas, renderer.camera);

let previousTime: number | undefined;
let accTime: number = 0;

createControlPanel(
  sceneOptions,
  simulation,
  renderer,
  cameraController,
  (options) => {
    const newScene = new Scene(options);

    simulation.reset(newScene);

    accTime = 0;
    previousTime = undefined;
  },
);

const frame = (time: number): void => {
  const dt = previousTime === undefined 
    ? 0 : Math.min((time - previousTime) * 0.001, 0.1);
  
  previousTime = time;

  const encoder = device.createCommandEncoder();

  if (gpuProfiling) {
    simulation.encodeStep(encoder);
  } else {
    accTime += dt;
    const fixedDt = scene.simulationConfig.fixedDt;

    while (accTime >= fixedDt) {
      simulation.encodeStep(encoder);
      accTime -= fixedDt;
    }
  }

  cameraController.update(dt);

  renderer.onFrame();
  renderer.encode(encoder);

  device.queue.submit([encoder.finish()]);
  
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
