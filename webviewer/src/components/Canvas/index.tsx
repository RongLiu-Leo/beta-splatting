import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { EngineContext } from '@/radix-sort/EngineContext'
import {
  Camera,
  CameraFileParser,
  InteractiveCamera,
} from '@/splatting-web/camera'
import { PackedGaussians, loadFileAsArrayBuffer } from '@/splatting-web/ply'
import { Renderer } from '@/splatting-web/renderer'
import { ChangeEvent, ReactNode, useEffect, useRef, useState } from 'react'
import { useStore } from 'zustand'
import { renderStore } from '../../splatting-web/index'
import { Slider } from '../ui/Slider'
import { Button } from '../ui/button'

type DefaultValues = {
  scale: number
  focalX: number
  focalY: number
}

interface CanvasProps {
  children?: ReactNode
}

export default function Canvas({ children }: CanvasProps) {
  const { renderer, camera } = useStore(renderStore)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cameraButtonRef = useRef<HTMLInputElement>(null)
  const plyButtonRef = useRef<HTMLInputElement>(null)
  const cameraListRef = useRef<HTMLUListElement>(null)
  const fpsCounterRef = useRef<HTMLLabelElement>(null)

  const [cameraFileParser, setCameraFileParser] = useState<CameraFileParser>()
  const [cameraPositions, setCameraPositions] = useState<Camera[]>([])
  const [selectedCameraPosition, setSelectedCameraPosition] = useState('')

  const [defaultValues, setDefaultValues] = useState<DefaultValues>()

  const [isLoading, setIsLoading] = useState(false)

  const initCanvas = () => {
    if (
      !canvasRef.current ||
      !cameraButtonRef.current ||
      !plyButtonRef.current ||
      !cameraListRef.current
    ) {
      return
    }

    const canvas = canvasRef.current

    const interactiveCamera = InteractiveCamera.default(canvas)

    const cameraFileParser = new CameraFileParser(
      canvas,
      interactiveCamera.setNewCamera.bind(interactiveCamera),
    )

    setCameraFileParser(cameraFileParser)
  }

  async function onFileLoad(arrayBuffer: ArrayBuffer) {
    const { renderer: currentRenderer } = renderStore.getState()

    if (
      !canvasRef.current ||
      !cameraButtonRef.current ||
      !plyButtonRef.current ||
      !cameraListRef.current ||
      !fpsCounterRef.current
    ) {
      return
    }

    const canvas = canvasRef.current
    const interactiveCamera = InteractiveCamera.default(canvas)
    const fpsCounter = fpsCounterRef.current

    if (currentRenderer) {
      await currentRenderer.destroy()
    }

    const gaussians = new PackedGaussians(arrayBuffer)

    try {
      const context = await Renderer.requestContext(gaussians)
      const engine_ctx = new EngineContext()
      await engine_ctx.initialize()
      const renderer = new Renderer(
        canvas,
        interactiveCamera,
        gaussians,
        context,
        fpsCounter,
      )

      renderStore.setState({ renderer })

      const camera = renderer.interactiveCamera.getCamera()
      setDefaultValues({
        scale: camera.scaleModifier,
        focalX: camera.focalX,
        focalY: camera.focalY,
      })
    } catch (error) {
      alert(error)
    }
  }

  async function handlePlyChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]

    if (file) {
      try {
        setIsLoading(true)
        const arrayBuffer = await loadFileAsArrayBuffer(file)
        onFileLoad(arrayBuffer)
      } catch (error) {
        alert(error)
      } finally {
        setIsLoading(false)
      }
    }
  }

  const handleReset = () => {
    if (!camera || !renderer || !defaultValues) return
    camera.setScale(defaultValues.scale)
    camera.setFocalX(defaultValues.focalX)
    camera.setFocalY(defaultValues.focalY)
    renderer.invalidate()
  }

  useEffect(() => {
    initCanvas()
  }, [])

  return (
    <div className="relative flex h-screen w-screen">
      <div className="absolute z-10 h-screen w-64 space-y-3 bg-white/20 px-2">
        <Label htmlFor="ply">Select a ply file</Label>
        <Input
          id="ply"
          ref={plyButtonRef}
          onChange={handlePlyChange}
          type="file"
          accept=".ply"
        />
        <Label htmlFor="cameraPos">Camera Position json</Label>
        <Input
          onChange={async (e) => {
            const cameraPositions =
              await cameraFileParser?.handleFileInputChange(e)
            if (!cameraPositions) return
            setCameraPositions(cameraPositions)
          }}
          ref={cameraButtonRef}
          type="file"
          accept="application/json"
        />
        <ScrollArea
          className={cn(
            'h-72 w-full rounded-md border',
            !cameraPositions.length && 'hidden',
          )}
        >
          <ul ref={cameraListRef} className="cameraList">
            {cameraPositions?.map((cameraPosition, index) => (
              <li
                key={index}
                className={cn(
                  'cursor-pointer p-1 text-xs hover:bg-gray-100',
                  selectedCameraPosition === cameraPosition.id.toString() &&
                    'bg-gray-200',
                )}
                onClick={() => {
                  setSelectedCameraPosition(cameraPosition.id.toString())
                  renderer?.interactiveCamera?.setNewCamera(cameraPosition)
                  setDefaultValues({
                    scale: cameraPosition.scaleModifier,
                    focalX: cameraPosition.focalX,
                    focalY: cameraPosition.focalY,
                  })
                }}
              >
                {index + 1}
              </li>
            ))}
          </ul>
        </ScrollArea>

        {renderer && (
          <>
            <Label htmlFor="scaleSlider">Gaussian scale</Label>
            <Slider
              id="scaleSlider"
              className="p-5"
              onValueChange={async (value) => {
                if (!camera || !renderer) return
                camera.setScale(value[0]!)
                renderer.invalidate()
              }}
              value={[camera?.scaleModifier || 0]}
              max={1}
              step={0.01}
              min={0}
            />

            <Button className="w-full" onClick={handleReset}>
              Reset
            </Button>
          </>
        )}
        <Button asChild variant="secondary" className="w-full">
          <a
            target="_blank"
            rel="noreferrer"
            href="https://drive.google.com/drive/folders/1WXCpR3kshQt2jmOtuCBsHKfzt1IMqey2?usp=sharing"
          >
            Download specific scene
          </a>
        </Button>
        <Button asChild variant="secondary" className="w-full">
          <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip">
            Download all scenes - 14GB
          </a>
        </Button>

        <div className="space-y-2 text-sm">
          <h2>Controls:</h2>
          <ul className="space-y-1">
            <li>Left click + drag: Look around</li>
            <li>W/S: Forward / Back</li>
            <li>A/D: Left / Right</li>
            <li>Q/E: Spin</li>
            <li>Z/X: Down / Up</li>
          </ul>
        </div>
      </div>

      <label
        ref={fpsCounterRef}
        className="absolute right-0 top-0 z-10 w-52 rounded bg-white/40 p-2"
      ></label>

      <canvas
        className="canvas-webgpu relative h-screen w-screen object-cover"
        ref={canvasRef}
        // height={600}
        // width={1200}
        // height={720}
        // width={1440}
        // height={800}
        // width={1600}
        height={960}
        width={1840}
      >
        {children}
      </canvas>

      {isLoading && (
        <div className="absolute left-0 top-0 flex h-full w-full items-center justify-center bg-white bg-opacity-50">
          <div className="h-32 w-32 animate-spin rounded-full border-b-2 border-t-2 border-gray-900"></div>
        </div>
      )}
    </div>
  )
}
