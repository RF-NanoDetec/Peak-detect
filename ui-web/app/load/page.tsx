"use client"

import { useState, useRef, useEffect } from "react"
import { toast } from "sonner"
import { useRouter } from "next/navigation"

import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { LoadControls } from "@/components/load/LoadControls"
import { LoadView } from "@/components/load/LoadView"

import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useProtocolStore } from "@/lib/stores/protocolStore"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { parseFilePreview, type LocalPreviewResult } from "@/lib/localDataParser"
import type { RecentSession } from "@/lib/types"

// File handling helpers
const sessionHandleCache = new Map<string, FileSystemFileHandle[]>()
const sessionFileCache = new Map<string, File[]>() 

const createSessionKey = (fileNames: string[]) =>
  fileNames.length ? [...fileNames].sort((a, b) => a.localeCompare(b)).join("|") : ""

const filesMatchGroup = (files: File[] | null, fileGroup: string[]) => {
  if (!files || files.length !== fileGroup.length) return false
  const selectedNames = files.map((file) => file.name)
  const expectedNames = [...fileGroup]
  selectedNames.sort((a, b) => a.localeCompare(b))
  expectedNames.sort((a, b) => a.localeCompare(b))
  return selectedNames.every((name, idx) => name === expectedNames[idx])
}

const handlesToFiles = async (handles: FileSystemFileHandle[]): Promise<File[]> => {
  return Promise.all(handles.map((handle) => handle.getFile()))
}

const cacheSessionFiles = (files: File[], handles?: FileSystemFileHandle[]) => {
  if (!files.length) return
  const key = createSessionKey(files.map((file) => file.name))
  if (!key) return
  sessionFileCache.set(key, files)
  if (handles) {
    sessionHandleCache.set(key, handles)
  }
}

const getRelativeDirectory = (file: File) => {
  const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath
  if (!relativePath) return null
  const pathParts = relativePath.split("/").filter(Boolean)
  if (pathParts.length <= 1) return null
  pathParts.pop()
  return pathParts.join("/")
}

const getCommonDirectory = (files: File[]) => {
  const directories = files.map(getRelativeDirectory).filter((dir): dir is string => Boolean(dir))
  if (directories.length !== files.length || directories.length === 0) return null
  const firstDirectory = directories[0]
  return directories.every((dir) => dir === firstDirectory) ? firstDirectory : null
}

const getSharedFilenamePattern = (fileNames: string[]) => {
  if (fileNames.length < 2) return null
  const stems = fileNames.map((name) => name.replace(/\.[^.]+$/, ""))
  const shortest = stems.reduce((a, b) => (a.length <= b.length ? a : b), stems[0])
  let prefixLength = 0
  while (
    prefixLength < shortest.length &&
    stems.every((stem) => stem[prefixLength]?.toLowerCase() === shortest[prefixLength].toLowerCase())
  ) {
    prefixLength += 1
  }

  const prefix = shortest
    .slice(0, prefixLength)
    .replace(/[\s._-]*\d*[\s._-]*$/, "")
    .trim()

  return prefix.length >= 4 ? prefix : null
}

const describeRecentSessionOrigin = (files: File[]): Pick<RecentSession, "originLabel" | "originKind"> => {
  const directory = getCommonDirectory(files)
  if (directory) {
    return {
      originLabel: directory,
      originKind: "folder",
    }
  }

  const filenamePattern = getSharedFilenamePattern(files.map((file) => file.name))
  if (filenamePattern) {
    return {
      originLabel: filenamePattern,
      originKind: "filename-pattern",
    }
  }

  return {
    originLabel: files.length === 1 ? files[0].name : "Local file selection",
    originKind: "local-selection",
  }
}

const describeLoadError = (error: any) => {
  if (error?.message === "Network Error" || error?.code === "ERR_NETWORK") {
    return "Cannot reach the local analysis service at http://127.0.0.1:8765. Start the backend with run_app.ps1 or python -m service.app, then try loading again."
  }

  if (error?.code === "ECONNABORTED") {
    return "Loading timed out. The file may be large or the local service may be busy."
  }

  if (error.response?.data) {
    const data = error.response.data
    if (typeof data.detail === 'string') {
      return data.detail
    }
    if (Array.isArray(data.detail)) {
      return data.detail.map((err: any) =>
        `${err.loc?.join('.')}: ${err.msg}`
      ).join(', ')
    }
    if (data.message) {
      return data.message
    }
  }

  return error?.message || 'Failed to load files'
}

const restoreCachedFiles = async (fileGroup: string[]): Promise<File[] | null> => {
  const key = createSessionKey(fileGroup)
  if (!key) return null

  const handles = sessionHandleCache.get(key)
  if (handles) {
    try {
      const files = await handlesToFiles(handles)
      if (filesMatchGroup(files, fileGroup)) {
        return files
      }
    } catch (error) {
      console.warn("Stored file handles are no longer valid:", error)
    }
    sessionHandleCache.delete(key)
  }

  const cachedFiles = sessionFileCache.get(key)
  if (cachedFiles && filesMatchGroup(cachedFiles, fileGroup)) {
    return cachedFiles
  }

  return null
}

export default function LoadDataPage() {
  const router = useRouter()
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [localPreview, setLocalPreview] = useState<LocalPreviewResult | null>(null)
  const previewTask = useRef(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { setResultId, setFiles, setMeta, setPreviewData, clearData, recentFiles, addRecentFiles } = useDataStore()
  const { params, updateParam } = useParamsStore()
  const { photonCorrection, protocol, setApplyCorrection, setDeadTimeNs, updateProtocol } = useProtocolStore()
  const { clearResults } = useResultsStore()
  
  // Time resolution conversion: display in milliseconds, store in seconds
  const timeResolutionMs = params.time_resolution * 1000
  const handleTimeResolutionChange = (ms: number) => {
    const seconds = ms / 1000
    updateParam('time_resolution', seconds)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      handleSelectedFiles(files)
    }
  }

  const handleSelectedFiles = (files: File[]) => {
    setSelectedFiles(files)
    toast.info(`Selected ${files.length} file(s)`)
    cacheSessionFiles(files)
  }

  const openFileSelectionFlow = async () => {
    if ('showOpenFilePicker' in window) {
      const files = await selectFilesWithHandles()
      if (files && files.length > 0) {
        handleSelectedFiles(files)
        return
      }
    }
    fileInputRef.current?.click()
  }
  
  const selectFilesWithHandles = async (): Promise<File[] | null> => {
    if (!('showOpenFilePicker' in window)) {
      return null
    }
    
    try {
      const handles = await (window as any).showOpenFilePicker({
        multiple: true,
        types: [{
          description: 'Data files',
          accept: {
            'text/plain': ['.txt'],
            'application/vnd.ms-excel': ['.xls'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
          }
        }]
      })
      
      const files = await handlesToFiles(handles)
      cacheSessionFiles(files, handles)
      return files
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Error selecting files:', error)
      }
      return null
    }
  }
  
  useEffect(() => {
    if (selectedFiles.length === 0) {
      setLocalPreview(null)
      setPreviewError(null)
      setPreviewLoading(false)
      previewTask.current += 1
      return
    }

    const file = selectedFiles[0]
    const taskId = ++previewTask.current
    setPreviewLoading(true)
    setPreviewError(null)

    parseFilePreview(file)
      .then((result) => {
        if (previewTask.current !== taskId) return
        setLocalPreview(result)
        setPreviewError(null)
      })
      .catch((error: any) => {
        if (previewTask.current !== taskId) return
        console.error('Local preview parsing failed:', error)
        setLocalPreview(null)
        setPreviewError(error?.message || 'Failed to parse selected file')
      })
      .finally(() => {
        if (previewTask.current !== taskId) return
        setPreviewLoading(false)
      })
  }, [selectedFiles])

  const handleLoadFiles = async () => {
    if (selectedFiles.length === 0) {
      toast.error("Please select files to load")
      return
    }

    setLoading(true)
    try {
      clearResults()
      clearData()
      
      const response = await apiClient.uploadFiles(selectedFiles, params.time_resolution, {
        applyDeadTimeCorrection: photonCorrection.applyCorrection,
        deadTimeNs: photonCorrection.deadTimeNs,
        protocol: protocol,
      })

      setResultId(response.resultId)
      setFiles(response.meta.files)
      setMeta(response.meta)
      
      const fileNames = selectedFiles.map(f => f.name)
      addRecentFiles(fileNames, describeRecentSessionOrigin(selectedFiles))
      cacheSessionFiles(selectedFiles)

      const previewData = await apiClient.getDataPreview(response.resultId)
      setPreviewData(previewData)

      toast.success(`Loaded ${response.meta.total_files} file(s) successfully`)
      
      setTimeout(() => router.push('/preprocess'), 500)
    } catch (error: any) {
      console.error('Failed to load files:', error)
      toast.error(describeLoadError(error))
    } finally {
      setLoading(false)
    }
  }

  const handleRecentFilesClick = async (session: RecentSession) => {
    const fileGroup = session.files
    const cachedFiles = await restoreCachedFiles(fileGroup)
    if (cachedFiles) {
      setSelectedFiles(cachedFiles)
      toast.success(`Restored ${cachedFiles.length} file(s) from recent session`)
      return
    }

    if ('showOpenFilePicker' in window) {
      toast.info(`Please re-select these files: ${fileGroup.join(', ')}`)
      const files = await selectFilesWithHandles()
      if (!files) {
        return
      }
      if (filesMatchGroup(files, fileGroup)) {
        setSelectedFiles(files)
        toast.success(`Selected ${files.length} file(s) from recent session`)
        return
      }
      toast.warning('Selected files do not match this session. Please try again.')
      return
    }

    toast.info(`Please select these files: ${fileGroup.join(', ')}`)
    fileInputRef.current?.click()
  }

  return (
    <PageShell>
      <PageVisualization>
        <LoadView
          selectedFiles={selectedFiles}
          loading={loading}
          previewLoading={previewLoading}
          previewError={previewError}
          localPreview={localPreview}
          protocol={protocol}
          updateProtocol={updateProtocol}
          onSelectFiles={openFileSelectionFlow}
          onDropFiles={handleSelectedFiles}
          onLoadFiles={handleLoadFiles}
          recentFiles={recentFiles}
          onRecentClick={handleRecentFilesClick}
        />
      </PageVisualization>
      <PageControls>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.xls,.xlsx"
          onChange={handleFileSelect}
          className="hidden"
        />
        <LoadControls
          timeResolutionMs={timeResolutionMs}
          onTimeResolutionChange={handleTimeResolutionChange}
          photonCorrection={photonCorrection}
          setApplyCorrection={setApplyCorrection}
          setDeadTimeNs={setDeadTimeNs}
        />
      </PageControls>
    </PageShell>
  )
}
