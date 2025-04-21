export class EngineContext {
  device: GPUDevice | null
  queue: GPUQueue | null
  cache: { [key: string]: any }

  constructor() {
    this.device = null
    this.queue = null
    // window.engine_ctx = this

    this.cache = {}
    this.cache.bindGroupLayouts = {}
    this.cache.pipelines = {}
  }

  async initialize() {
    if (this.device != null) return
    const entry = navigator.gpu
    if (!entry) {
      throw new Error('WebGPU is not supported')
    }
    const adapter = await entry.requestAdapter()

    if (!adapter) throw new Error('No adapter found')

    this.device = await adapter.requestDevice()
    this.queue = this.device.queue
  }

  createBuffer0(size: number, usage: number) {
    const desc = {
      size: (size + 3) & ~3,
      usage,
    }

    if (!this.device) throw new Error('Device not initialized')

    const buffer_out = this.device.createBuffer(desc)
    return buffer_out
  }

  createBuffer(
    buffer_in: globalThis.BufferSource,
    usage: number,
    offset = 0,
    size = buffer_in.byteLength,
  ) {
    if (!this.queue) throw new Error('Queue not initialized')
    if (!this.device) throw new Error('Device not initialized')

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
