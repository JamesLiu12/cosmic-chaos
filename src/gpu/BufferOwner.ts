export default class BufferOwner {
  private readonly device: GPUDevice;
  private readonly buffers: GPUBuffer[] = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  public createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.device.createBuffer(descriptor);
    this.buffers.push(buffer);
    return buffer;
  }

  public destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers.length = 0;
  }
}
