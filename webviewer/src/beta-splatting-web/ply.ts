import { PackingType, StaticArray, Struct, f32, vec3, vec4 } from './packing'

export function loadFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  /* loads a file as an ArrayBuffer (i.e. a binary blob) */
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (event) => {
      if (!event.target || !event.target.result) {
        throw new Error('Failed to load file')
      }
      if (typeof event.target.result === 'string') {
        throw new Error('Got a text file instead of a binary one')
      }
      resolve(event.target.result)
    }

    reader.onerror = (event) => {
      if (!event.target) {
        throw new Error('Failed to load file')
      }
      reject(event.target.error)
    }

    reader.readAsArrayBuffer(file)
  })
}

export class PackedBetas {
  /* A class that
        1) reads the binary blob from a .ply file
        2) converts internally into a structured representation
        3) packs the structured representation into a flat array of bytes as expected by the shaders
    */
  numBetas: number

  betaLayout: PackingType
  public betaArrayLayout: PackingType
  positionsLayout: PackingType
  public positionsArrayLayout: PackingType

  betasBuffer: ArrayBuffer
  positionsBuffer: ArrayBuffer

  constructor(arrayBuffer: ArrayBuffer) {
    // decode the header
    const [vertexCount, propertyTypes, vertexData] =
      PackedBetas.decodeHeader(arrayBuffer)
    this.numBetas = vertexCount

    // define the layout of a single point
    this.betaLayout = new Struct([
      ['position', new vec3(f32)],
      ['log_scale', new vec3(f32)],
      ['rot', new vec4(f32)],
      ['c0', new vec3(f32)],
      ['opacity_logit', f32],
      ['beta', f32],
      ['sb_params', new StaticArray(new vec3(f32), 4)], // Support for 2 spherical beta primitives (2 * 2 = 4 vec3s)
    ])
    
    // define the layout of the entire point cloud
    this.betaArrayLayout = new StaticArray(this.betaLayout, vertexCount)

    this.positionsLayout = new vec3(f32)
    this.positionsArrayLayout = new StaticArray(
      this.positionsLayout,
      vertexCount,
    )

    // pack the points
    this.betasBuffer = new ArrayBuffer(this.betaArrayLayout.size)
    const betaWriteView = new DataView(this.betasBuffer)

    this.positionsBuffer = new ArrayBuffer(this.positionsArrayLayout.size)
    const positionsWriteView = new DataView(this.positionsBuffer)

    let readOffset = 0
    let betaWriteOffset = 0
    let positionWriteOffset = 0
    for (let i = 0; i < vertexCount; i++) {
      const [newReadOffset, rawVertex] = this.readRawVertex(
        readOffset,
        vertexData,
        propertyTypes,
      )

      readOffset = newReadOffset
      betaWriteOffset = this.betaLayout.pack(
        betaWriteOffset,
        this.arrangeVertex(rawVertex),
        betaWriteView,
      )

      positionWriteOffset = this.positionsLayout.pack(
        positionWriteOffset,
        [rawVertex.x, rawVertex.y, rawVertex.z],
        positionsWriteView,
      )
    }
  }

  private static decodeHeader(
    plyArrayBuffer: ArrayBuffer,
  ): [number, Record<string, string>, DataView] {
    /* decodes the .ply file header and returns a tuple of:
     * - vertexCount: number of vertices in the point cloud
     * - propertyTypes: a map from property names to their types
     * - vertexData: a DataView of the vertex data
     */

    const decoder = new TextDecoder()
    let headerOffset = 0
    let headerText = ''

    while (true) {
      const headerChunk = new Uint8Array(plyArrayBuffer, headerOffset, 50)
      headerText += decoder.decode(headerChunk)
      headerOffset += 50

      if (headerText.includes('end_header')) {
        break
      }
    }

    const headerLines = headerText.split('\n')

    let vertexCount = 0
    const propertyTypes: Record<string, string> = {}

    for (let i = 0; i < headerLines.length; i++) {
      const line = headerLines[i]!.trim()
      if (line.startsWith('element vertex')) {
        const vertexCountMatch = line.match(/\d+/)
        if (vertexCountMatch) {
          vertexCount = parseInt(vertexCountMatch[0]!)
        }
      } else if (line.startsWith('property')) {
        const propertyMatch = line.match(/(\w+)\s+(\w+)\s+(\w+)/)
        if (propertyMatch) {
          const propertyType = propertyMatch[2]!
          const propertyName = propertyMatch[3]!
          propertyTypes[propertyName] = propertyType
        }
      } else if (line === 'end_header') {
        break
      }
    }

    const vertexByteOffset =
      headerText.indexOf('end_header') + 'end_header'.length + 1
    const vertexData = new DataView(plyArrayBuffer, vertexByteOffset)

    return [vertexCount, propertyTypes, vertexData]
  }

  private readRawVertex(
    offset: number,
    vertexData: DataView,
    propertyTypes: Record<string, string>,
  ): [number, Record<string, number>] {
    /* reads a single vertex from the vertexData DataView and returns a tuple of:
     * - offset: the offset of the next vertex in the vertexData DataView
     * - rawVertex: a map from property names to their values
     */
    const rawVertex: Record<string, number> = {}

    for (const property in propertyTypes) {
      const propertyType = propertyTypes[property]
      if (propertyType === 'float') {
        rawVertex[property] = vertexData.getFloat32(offset, true)
        offset += Float32Array.BYTES_PER_ELEMENT
      } else if (propertyType === 'uchar') {
        rawVertex[property] = vertexData.getUint8(offset) / 255.0
        offset += Uint8Array.BYTES_PER_ELEMENT
      }
    }

    return [offset, rawVertex]
  }

  private arrangeVertex(
    rawVertex: Record<string, number>,
  ): Record<string, any> {
    /* arranges a raw vertex into a vertex that can be packed by the betaLayout utility */
    
    // Extract beta parameter if it exists, otherwise default to 0
    const beta = rawVertex.beta !== undefined ? rawVertex.beta : 0.0;
    
    // Extract spherical beta parameters if they exist
    const sbParams = [];
    let sbIndex = 0;
    while (true) {
      const colorPrefix = `sb_params_${sbIndex * 6}`;
      const dirPrefix = `sb_params_${sbIndex * 6 + 3}`;
      
      // Check if this spherical beta primitive exists
      if (rawVertex[colorPrefix] === undefined) break;
      
      // Extract color (r,g,b)
      sbParams.push([
        rawVertex[colorPrefix],
        rawVertex[colorPrefix + 1],
        rawVertex[colorPrefix + 2]
      ]);
      
      // Extract direction parameters (theta, phi, beta)
      sbParams.push([
        rawVertex[dirPrefix],
        rawVertex[dirPrefix + 1],
        rawVertex[dirPrefix + 2]
      ]);
      
      sbIndex++;
    }

    const arrangedVertex: Record<string, any> = {
      position: [rawVertex.x, rawVertex.y, rawVertex.z],
      logScale: [rawVertex.scale_0, rawVertex.scale_1, rawVertex.scale_2],
      rotQuat: [
        rawVertex.rot_0,
        rawVertex.rot_1,
        rawVertex.rot_2,
        rawVertex.rot_3,
      ],
      opacityLogit: rawVertex.opacity,
      beta,
      sbParams
    }
    return arrangedVertex
  }
}
