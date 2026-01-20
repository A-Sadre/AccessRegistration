export interface PersonalInfo {
  firstName: string
  lastName: string
  documentNumber: string
  nationality: string
  dateOfBirth: string
  sex: string
  expirationDate: string
}

export interface ExtractedResult {
  info: PersonalInfo
  faceDataUrl?: string
}
