export type ComputePass = {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly label: string;
};

type ComputePassDescriptor = {
  label: string;
  compute: GPUProgrammableStage;
  buffers: Readonly<Record<number, GPUBuffer>>;
  bindGroupLabel?: string;
  passLabel?: string;
};

export function createComputePass(device: GPUDevice, descriptor: ComputePassDescriptor): ComputePass {
  const { label, compute, buffers, bindGroupLabel, passLabel = label } = descriptor;
  const pipeline = device.createComputePipeline({ label, layout: "auto", compute });
  const bindGroup = device.createBindGroup({
    label: bindGroupLabel,
    layout: pipeline.getBindGroupLayout(0),
    entries: Object.entries(buffers).map(([binding, buffer]) => ({
      binding: Number(binding),
      resource: { buffer },
    })),
  });
  return { pipeline, bindGroup, label: passLabel };
}

export function encodeComputePass(
  encoder: GPUCommandEncoder,
  step: ComputePass,
  dispatch: number | GPUBuffer,
  offset = 0,
): void {
  const pass = encoder.beginComputePass({ label: step.label });
  pass.setPipeline(step.pipeline);
  pass.setBindGroup(0, step.bindGroup);
  if (typeof dispatch === "number") {
    pass.dispatchWorkgroups(dispatch);
  } else {
    pass.dispatchWorkgroupsIndirect(dispatch, offset);
  }
  pass.end();
}
