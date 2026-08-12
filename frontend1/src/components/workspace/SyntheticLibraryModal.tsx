import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertCircle, Check, FlaskConical, Loader2, Plus, Search } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { getMeta, getStoreDocuments } from '@/api/studio'
import { useImportStatus, useSyntheticImportStore } from '@/store/syntheticImportStore'
import type { StoreDocument } from '@/types/studio'

interface SyntheticLibraryModalProps {
  workspaceId: string
  workspaceName: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

function LibraryRow({
  doc,
  workspaceId,
  workspaceName,
}: {
  doc: StoreDocument
  workspaceId: string
  workspaceName: string
}) {
  const status = useImportStatus(workspaceId, doc.id)
  const alreadyImported = useMemo(() => {
    if (!Array.isArray(doc.imported_into)) return false
    return doc.imported_into.some((entry) =>
      typeof entry === 'string'
        ? entry === workspaceId
        : entry?.workspace_id === workspaceId
    )
  }, [doc.imported_into, workspaceId])

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2.5 dark:border-border-dark">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-ink dark:text-ink-inverted">{doc.title}</p>
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          <Badge variant="secondary">{doc.doc_type}</Badge>
          {doc.industry && <Badge variant="outline">{doc.industry}</Badge>}
          {doc.language && <Badge variant="outline">{doc.language}</Badge>}
        </div>
      </div>
      <div className="shrink-0">
        {status === 'importing' ? (
          <Button variant="outline" size="sm" disabled>
            <Loader2 className="h-4 w-4 animate-spin" />
            Adding…
          </Button>
        ) : status === 'done' || alreadyImported ? (
          <Button variant="outline" size="sm" disabled>
            <Check className="h-4 w-4" />
            Added
          </Button>
        ) : status === 'error' ? (
          <Button
            variant="outline"
            size="sm"
            className="text-danger-600"
            onClick={() => useSyntheticImportStore.getState().startImport(workspaceId, workspaceName, doc.id, doc.title)}
          >
            <AlertCircle className="h-4 w-4" />
            Retry
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => useSyntheticImportStore.getState().startImport(workspaceId, workspaceName, doc.id, doc.title)}
          >
            <Plus className="h-4 w-4" />
            Add to workspace
          </Button>
        )}
      </div>
    </div>
  )
}

export function SyntheticLibraryModal({ workspaceId, workspaceName, open, onOpenChange }: SyntheticLibraryModalProps) {
  const [docType, setDocType] = useState('all')
  const [search, setSearch] = useState('')

  const { data: meta } = useQuery({
    queryKey: ['studio', 'meta'],
    queryFn: () => getMeta(),
    enabled: open,
  })
  const docTypeOptions = meta?.doc_types ?? []

  const { data, isLoading } = useQuery({
    queryKey: ['studio', 'store-documents', docType],
    queryFn: () => getStoreDocuments(docType === 'all' ? undefined : docType),
    enabled: open,
  })

  const documents = useMemo(() => {
    const all = data?.documents ?? []
    const q = search.trim().toLowerCase()
    if (!q) return all
    return all.filter((doc) => doc.title.toLowerCase().includes(q))
  }, [data, search])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] max-w-3xl flex-col">
        <DialogHeader>
          <DialogTitle>Synthetic Library</DialogTitle>
          <DialogDescription>
            Add published synthetic documents from the Synthetic Data Studio into this workspace.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-subtle" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by title…"
              className="pl-8"
            />
          </div>
          <Select value={docType} onValueChange={setDocType}>
            <SelectTrigger className="w-56 shrink-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All document types</SelectItem>
              {docTypeOptions.map((dt) => (
                <SelectItem key={dt} value={dt}>
                  {dt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : documents.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center">
            <FlaskConical className="h-8 w-8 text-ink-subtle" />
            <p className="text-sm font-medium text-ink dark:text-ink-inverted">
              {search ? 'No documents match your search' : 'No synthetic documents found'}
            </p>
            <p className="text-xs text-ink-subtle">
              {search ? 'Try a different title or clear the search.' : 'Publish documents from the Synthetic Data Studio to see them here.'}
            </p>
          </div>
        ) : (
          <ScrollArea className="flex-1 min-h-[24rem]">
            <div className="space-y-2 pr-3">
              {documents.map((doc) => (
                <LibraryRow key={doc.id} doc={doc} workspaceId={workspaceId} workspaceName={workspaceName} />
              ))}
            </div>
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  )
}
