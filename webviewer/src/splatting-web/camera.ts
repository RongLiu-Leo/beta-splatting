import { ChangeEvent } from 'react'
import { Mat3, Mat4, Vec3, mat3, mat4, vec3 } from 'wgpu-matrix'
import { renderStore } from '.'

// camera as loaded from JSON
interface CameraRaw {
  id: number
  img_name: string
  width: number
  height: number
  position: number[]
  rotation: number[][]
  fx: number
  fy: number
}

// for some reason this needs to be a bit different than the one in wgpu-matrix
function getProjectionMatrix(
  znear: number,
  zfar: number,
  fovX: number,
  fovY: number,
): Mat4 {
  const tanHalfFovY: number = Math.tan(fovY / 2)
  const tanHalfFovX: number = Math.tan(fovX / 2)

  const top: number = tanHalfFovY * znear
  const bottom: number = -top
  const right: number = tanHalfFovX * znear
  const left: number = -right

  const P: Mat4 = mat4.create()

  const z_sign: number = 1.0

  P[0] = (2.0 * znear) / (right - left)
  P[5] = (2.0 * znear) / (top - bottom)
  P[8] = (right + left) / (right - left)
  P[9] = (top + bottom) / (top - bottom)
  P[10] = (z_sign * zfar) / (zfar - znear)
  P[11] = -(zfar * znear) / (zfar - znear)
  P[14] = z_sign
  P[15] = 0.0

  return mat4.transpose(P)
}

// A camera as used by the renderer. Interactivity is handled by InteractiveCamera.
export class Camera {
  height: number
  width: number
  viewMatrix: Mat4
  perspective: Mat4
  focalX: number
  focalY: number
  scaleModifier: number
  id: number

  constructor(
    height: number,
    width: number,
    viewMatrix: Mat4,
    perspective: Mat4,
    focalX: number,
    focalY: number,
    scaleModifier: number,
  ) {
    console.log(`Camera: ${height}x${width}`)

    this.id = Math.ceil(Math.random() * 10000)

    this.height = height
    this.width = width
    this.viewMatrix = viewMatrix
    this.perspective = perspective
    this.focalX = focalX
    this.focalY = focalY
    this.scaleModifier = scaleModifier
  }

  static default(): Camera {
    const canvasW = 960
    const canvasH = 960

    const fovFactor = 1

    const fovX = focal2fov(canvasW / 2, canvasW) / fovFactor
    const fovY = focal2fov(canvasH, canvasH) / fovFactor

    const projectionMatrix = getProjectionMatrix(0.2, 10, fovX, fovY)

    const viewMatrix = mat4.create(
      0.582345724105835,
      -0.3235852122306824,
      0.7372694611549377,
      0,
      0.23868794739246368,
      0.9381394982337952,
      0.22253619134426117,
      0,
      -0.7680802941322327,
      0.04477229341864586,
      0.6242981553077698,
      0,
      0.13517332077026367,
      -1.1848870515823364,
      3.3873789310455322,
      1,
    )

    return new Camera(
      canvasW,
      canvasH,
      // 1,
      // 1,
      viewMatrix,
      projectionMatrix,
      // fovX,
      // fovY,
      canvasW,
      canvasH,
      1 * fovFactor,
    )
  }

  public setScale(scale: number) {
    this.scaleModifier = scale
  }

  public setFocalX(focalX: number) {
    this.focalX = focalX
  }

  public setFocalY(focalY: number) {
    this.focalY = focalY
  }

  // computes the depth of a point in camera space, for sorting
  dotZ(): (v: Vec3) => number {
    const depthAxis = this.depthAxis()
    return (v: Vec3) => {
      return vec3.dot(depthAxis, v)
    }
  }

  // gets the camera position in world space, for evaluating the spherical harmonics
  getPosition(): Vec3 {
    const inverseViewMatrix = mat4.inverse(this.viewMatrix)
    return mat4.getTranslation(inverseViewMatrix)
  }

  getProjMatrix(): Mat4 {
    return mat4.multiply(this.perspective, this.viewMatrix)
  }

  // for camera interactions
  translate(x: number, y: number, z: number) {
    const viewInv = mat4.inverse(this.viewMatrix)
    mat4.translate(viewInv, [x, y, z], viewInv)
    mat4.inverse(viewInv, this.viewMatrix)
  }

  // for camera interactions
  rotate(x: number, y: number, z: number) {
    const viewInv = mat4.inverse(this.viewMatrix)
    mat4.rotateX(viewInv, y, viewInv)
    mat4.rotateY(viewInv, x, viewInv)
    mat4.rotateZ(viewInv, z, viewInv)
    mat4.inverse(viewInv, this.viewMatrix)
  }

  isInsideFrustum(point: Vec3) {
    const projMatrix = this.getProjMatrix()
    const p = vec3.transformMat4(point, projMatrix, [0, 0, 0])
    return (
      p[0]! >= -1 &&
      p[0]! <= 1 &&
      p[1]! >= -1 &&
      p[1]! <= 1 &&
      p[2]! >= -1 &&
      p[2]! <= 1
    )
  }

  // the depth axis is the third column of the transposed view matrix
  private depthAxis(): Vec3 {
    return mat4.getAxis(mat4.transpose(this.viewMatrix), 2)
  }
}

// Adds interactivity to a camera. The camera is modified by the user's mouse and keyboard input.
export class InteractiveCamera {
  private camera: Camera
  private canvas: HTMLCanvasElement

  private drag: boolean = false
  private oldX: number = 0
  private oldY: number = 0
  private dRX: number = 0
  private dRY: number = 0
  private dRZ: number = 0
  private dTX: number = 0
  private dTY: number = 0
  private dTZ: number = 0

  private dirty: boolean = true

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera
    this.canvas = canvas

    this.createCallbacks()
  }

  static default(canvas: HTMLCanvasElement): InteractiveCamera {
    return new InteractiveCamera(Camera.default(), canvas)
  }

  private createCallbacks() {
    this.canvas.addEventListener(
      'mousedown',
      (e) => {
        this.drag = true
        this.oldX = e.pageX
        this.oldY = e.pageY
        this.setDirty()
        // await
        this.canvas.requestPointerLock()
        e.preventDefault()
      },
      false,
    )

    this.canvas.addEventListener(
      'mouseup',
      (e) => {
        this.drag = false
        document.exitPointerLock()
      },
      false,
    )

    this.canvas.addEventListener(
      'mousemove',
      (e) => {
        if (!this.drag) return false

        this.dRX = (e.movementX * 2 * Math.PI) / this.canvas.width
        this.dRY = (-e.movementY * 2 * Math.PI) / this.canvas.height
        this.oldX = e.pageX
        this.oldY = e.pageY
        this.setDirty()
        e.preventDefault()
      },
      false,
    )

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        this.dTZ = e.deltaY * 0.1
        this.setDirty()
        e.preventDefault()
      },
      false,
    )

    window.addEventListener(
      'keydown',
      (e) => {
        const keyMap: { [key: string]: () => void } = {
          // translation
          w: () => {
            this.dTZ += 0.1
          },
          s: () => {
            this.dTZ -= 0.1
          },
          a: () => {
            this.dTX -= 0.1
          },
          d: () => {
            this.dTX += 0.1
          },
          q: () => {
            this.dRZ -= 0.1
          },
          e: () => {
            this.dRZ += 0.1
          },
          z: () => {
            this.dTY += 0.1
          },
          x: () => {
            this.dTY -= 0.1
          },

          // rotation
          j: () => {
            this.dRX -= 0.1
          },
          l: () => {
            this.dRX += 0.1
          },
          i: () => {
            this.dRY += 0.1
          },
          k: () => {
            this.dRY -= 0.1
          },
          u: () => {
            this.dRZ -= 0.1
          },
          o: () => {
            this.dRZ += 0.1
          },
        }

        if (keyMap[e.key]) {
          keyMap[e.key]?.()
          this.setDirty()
          e.preventDefault()
        }
      },
      false,
    )
  }

  public setNewCamera(newCamera: Camera) {
    this.camera = newCamera
    this.setDirty()
  }

  private setDirty() {
    this.dirty = true
  }

  private setClean() {
    this.dirty = false
  }

  public isDirty(): boolean {
    return this.dirty
  }

  public getCamera(): Camera {
    if (this.isDirty()) {
      this.camera.translate(this.dTX, this.dTY, this.dTZ)
      this.camera.rotate(this.dRX, this.dRY, this.dRZ)
      this.dTX = this.dTY = this.dTZ = this.dRX = this.dRY = this.dRZ = 0
      this.setClean()
    }

    renderStore.setState({ camera: this.camera })

    return this.camera
  }
}

function focal2fov(focal: number, pixels: number): number {
  return 2 * Math.atan(pixels / (2 * focal))
}

function worldToCamFromRT(R: Mat3, t: Vec3): Mat4 {
  const R_ = R
  const camToWorld = mat4.fromMat3(R_)
  const minusT = vec3.mulScalar(t, -1)
  mat4.translate(camToWorld, minusT, camToWorld)
  return camToWorld
}

// converting camera coordinate systems is always black magic :(
function cameraFromJSON(
  rawCamera: CameraRaw,
  canvasW: number,
  canvasH: number,
): Camera {
  const fovX = focal2fov(960 / 2, 960)
  const fovY = focal2fov(960, 960)

  const projectionMatrix = getProjectionMatrix(0.2, 100, fovX, fovY)

  const R = mat3.create(...rawCamera.rotation.flat())
  // const T = rawCamera.position
  const T = new Float32Array(16)
  T.set(rawCamera.position)

  const viewMatrix = worldToCamFromRT(R, T)

  const camera = new Camera(960, 960, viewMatrix, projectionMatrix, 960, 960, 1)

  return camera
}

// A UI component that parses a JSON file containing a list of cameras and displays them as a list,
// allowing the user to choose from presets.
export class CameraFileParser {
  private currentLineId: number = 0
  private canvas: HTMLCanvasElement
  private cameraSetCallback: (camera: Camera) => void

  constructor(
    canvas: HTMLCanvasElement,
    cameraSetCallback: (camera: Camera) => void,
  ) {
    this.cameraSetCallback = cameraSetCallback

    this.canvas = canvas
  }

  public handleFileInputChange = async (e: ChangeEvent<HTMLInputElement>) => {
    return new Promise((resolve) => {
      const file = e.target.files?.[0]

      if (file) {
        const reader = new FileReader()
        reader.onload = (e) => {
          const cameras = this.handleFileLoad(e)
          resolve(cameras)
        }

        reader.readAsText(file)
      }
    }) as Promise<Camera[]>
  }

  public handleFileLoad = (event: ProgressEvent<FileReader>) => {
    if (!event.target) return

    const contents = event.target.result as string
    const jsonData = JSON.parse(contents)

    this.currentLineId = 0

    const cameras = jsonData.map((cameraJSON: any) => {
      this.currentLineId++

      const camera = cameraFromJSON(
        cameraJSON,
        this.canvas.width,
        this.canvas.height,
      )

      return camera
    })

    return cameras
  }

  public createCallbackForLine = (camera: Camera) => {
    return () => {
      this.cameraSetCallback(camera)
    }
  }
}
