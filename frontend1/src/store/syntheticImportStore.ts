import { create } from 'zustand'
import { importSyntheticDocument } from '@/api/workspaces'
import { runTrackedPipelineJob } from '@/lib/pipelineJobRunner'

interface ImportJob {
  storeDocumentId: string
  workspaceId: string
  status: 'importing' | 'done' | 'error'
}

interface SyntheticImportState {
  jobs: Record<string, ImportJob>
  startImport: (workspaceId: string, workspaceName: string, storeDocumentId: string, title: string) => void
}

export const useSyntheticImportStore = create<SyntheticImportState>((set, get) => ({
  jobs: {},

  startImport: (workspaceId, workspaceName, storeDocumentId, title) => {
    const jobKey = `${workspaceId}:${storeDocumentId}`
    if (get().jobs[jobKey]?.status === 'importing') return

    set((state) => ({
      jobs: { ...state.jobs, [jobKey]: { storeDocumentId, workspaceId, status: 'importing' } },
    }))

    // Runs through the exact same job-tracking path a manual upload does — it
    // shows up as a Run History card in the Ingest tab with real step-by-step
    // progress (Parse/Extract/Graph/Vector/Coverage), not a bespoke modal spinner
    // with no visible pipeline. See lib/pipelineJobRunner.ts.
    runTrackedPipelineJob({
      workspaceId,
      workspaceName,
      fileNames: [title],
      startMessage: `Importing "${title}" from Synthetic Library…`,
      stream: (onEvent, signal) => importSyntheticDocument(workspaceId, storeDocumentId, onEvent, signal),
    }).then((status) => {
      set((state) => ({
        jobs: { ...state.jobs, [jobKey]: { storeDocumentId, workspaceId, status } },
      }))
    })
  },
}))

export const useImportStatus = (workspaceId: string, storeDocumentId: string) =>
  useSyntheticImportStore((state) => state.jobs[`${workspaceId}:${storeDocumentId}`]?.status)
