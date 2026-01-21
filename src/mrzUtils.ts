import type { PersonalInfo } from './types'
import { apiUrl } from "./api";

export type ExtractResult = {
  info: PersonalInfo | null
  faceDataUrl?: string
  faceBox?: { x1: number; y1: number; x2: number; y2: number }
  mrzLines?: string[]
  mrzChecks?: Record<string, boolean>
}

export async function extractPersonalInfoFromImage(
  dataUrl: string,
): Promise<ExtractResult> {
  const blob = await (await fetch(dataUrl)).blob()
  const form = new FormData()
  form.append('file', blob, 'passport.jpg')

  const res = await fetch(apiUrl("/mrz/extract"), {

    method: 'POST',
    body: form,
  })

  const json = await res.json()
  if (!json?.ok) return { info: null }

  const f = json.fields || {}
  const info: PersonalInfo = {
    firstName: f.firstName || '',
    lastName: f.lastName || '',
    documentNumber: f.documentNumber || '',
    nationality: f.nationality || '',
    dateOfBirth: f.dateOfBirth || '',
    sex: f.sex || '',
    expirationDate: f.expirationDate || '',
  }

  const face = json.face
  const faceDataUrl =
    face?.ok && face?.b64_jpeg
      ? `data:image/jpeg;base64,${face.b64_jpeg}`
      : undefined

  return {
    info,
    faceDataUrl,
    faceBox: face?.box,
    mrzLines: json?.mrz?.lines,
    mrzChecks: json?.mrz?.checks,
  }
}

