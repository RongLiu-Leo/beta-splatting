import { Camera } from './camera'
import { Renderer } from './renderer'

import { createStore } from 'zustand'

interface RendererState {
  renderer: undefined | Renderer
  camera: undefined | Camera
}

export const renderStore = createStore<RendererState>(() => ({
  renderer: undefined,
  camera: undefined,
}))
