export class GpuContext {
  gpu: GPU
  adapter: GPUAdapter
  device: GPUDevice

  queue: GPUQueue
  cache: { [key: string]: any }
  hasTimestampQuery: boolean

  constructor(gpu: GPU, adapter: GPUAdapter, device: GPUDevice) {
    this.gpu = gpu
    this.adapter = adapter
    this.device = device

    this.queue = this.device.queue

    this.cache = {}
    this.cache.bindGroupLayouts = {}
    this.cache.pipelines = {}

    this.hasTimestampQuery = this.device.features.has('timestamp-query')
  }

  destroy() {
    this.device.destroy()
    this.adapter = null as any
    this.device = null as any
  }

  createBuffer0(size: number, usage: number) {
    const desc = {
      size: (size + 3) & ~3,
      usage,
    }

    const buffer_out = this.device.createBuffer(desc)
    return buffer_out
  }

  createBuffer(
    buffer_in: globalThis.BufferSource,
    usage: number,
    offset = 0,
    size = buffer_in.byteLength,
  ) {
    usage |= GPUBufferUsage.COPY_DST

    const desc: globalThis.GPUBufferDescriptor = {
      size: (size + 3) & ~3,
      usage,
    }
    const buffer_out = this.device.createBuffer(desc)
    this.queue.writeBuffer(buffer_out, 0, buffer_in, offset, size)

    return buffer_out
  }
}
