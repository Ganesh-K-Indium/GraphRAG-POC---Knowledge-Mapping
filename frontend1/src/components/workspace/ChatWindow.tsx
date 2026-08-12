import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useMutation } from '@tanstack/react-query'
import { MessageCircle, Send, ExternalLink, X, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { cn } from '@/lib/utils'
import { QUERY_TYPE_LABELS } from '@/lib/domain-taxonomy'
import { askQuestion } from '@/api/chat'
import type { ChatMessage } from '@/types/analysis'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { SourcesSection } from './EvidenceCards'

interface ChatWindowProps {
  workspaceId: string
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export function ChatWindow({ workspaceId }: ChatWindowProps) {
  // Disabled floating chat widget for demo video
  return null
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const mutation = useMutation({
    mutationFn: (question: string) => askQuestion(workspaceId, question),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: 'assistant',
          content: data.answer,
          queryType: data.query_type,
          evidence: data.evidence,
          timestamp: Date.now(),
        },
      ])
      requestAnimationFrame(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }))
    },
  })

  const handleSend = () => {
    const question = input.trim()
    if (!question || mutation.isPending) return
    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: 'user', content: question, timestamp: Date.now() },
    ])
    setInput('')
    mutation.mutate(question)
    requestAnimationFrame(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }))
  }

  return (
    <>
      <div className="fixed bottom-6 right-6 z-50">
        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.92, y: 12 }}
              transition={{ duration: 0.16 }}
              className="absolute bottom-16 right-0 flex h-[32rem] w-96 flex-col overflow-hidden rounded-lg border border-border bg-surface shadow-popover dark:border-border-dark dark:bg-surface-dark-subtle"
            >
              <div className="flex items-center justify-between border-b border-border px-4 py-3 dark:border-border-dark">
                <span className="text-sm font-semibold text-ink dark:text-ink-inverted">Quick Q&amp;A</span>
                <div className="flex items-center gap-1">
                  <Link
                    to={`/workspace/${workspaceId}/chat`}
                    className="rounded-md p-1.5 text-ink-muted hover:bg-surface-muted hover:text-ink dark:text-ink-subtle dark:hover:bg-surface-dark-muted dark:hover:text-ink-inverted"
                    title="Open full chat"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </Link>
                  <button
                    type="button"
                    onClick={() => setOpen(false)}
                    className="rounded-md p-1.5 text-ink-muted hover:bg-surface-muted hover:text-ink dark:text-ink-subtle dark:hover:bg-surface-dark-muted dark:hover:text-ink-inverted"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="flex-1 space-y-3 overflow-y-auto p-4">
                {messages.length === 0 && (
                  <p className="text-sm text-ink-subtle">
                    Ask a quick question about this workspace&apos;s contracts, coverage, or risks.
                  </p>
                )}
                {messages.map((m) => (
                  <div key={m.id} className={cn('flex flex-col gap-1', m.role === 'user' ? 'items-end' : 'items-start')}>
                    <div
                      className={cn(
                        'max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed',
                        m.role === 'user'
                          ? 'bg-accent-600 text-white'
                          : 'bg-surface-muted text-ink dark:bg-surface-dark-subtle dark:text-ink-inverted'
                      )}
                    >
                      {m.role === 'assistant' ? (
                        <div className="[&_p]:my-1.5 [&_ul]:my-1.5 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:my-1.5 [&_ol]:list-decimal [&_ol]:pl-4 [&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs [&_th]:border [&_th]:border-border [&_th]:bg-surface-subtle [&_th]:px-1.5 [&_th]:py-1 [&_th]:text-left [&_th]:font-semibold dark:[&_th]:border-border-dark dark:[&_th]:bg-surface-dark-muted [&_td]:border [&_td]:border-border [&_td]:px-1.5 [&_td]:py-1 dark:[&_td]:border-border-dark">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                      ) : (
                        m.content
                      )}
                    </div>
                    {m.role === 'assistant' && m.queryType && (
                      <Badge variant="secondary" className="text-[10px]">
                        {QUERY_TYPE_LABELS[m.queryType]}
                      </Badge>
                    )}
                    {m.role === 'assistant' && m.evidence && m.evidence.length > 0 && (
                      <div className="w-full max-w-[95%]">
                        <SourcesSection evidence={m.evidence} />
                      </div>
                    )}
                  </div>
                ))}
                {mutation.isPending && (
                  <div className="flex items-center gap-2 text-sm text-ink-subtle">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Thinking…
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              <div className="flex items-end gap-2 border-t border-border p-3 dark:border-border-dark">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSend()
                    }
                  }}
                  placeholder="Ask a question…"
                  className="min-h-[40px] resize-none text-sm"
                  rows={1}
                />
                <Button
                  size="icon"
                  onClick={handleSend}
                  disabled={!input.trim() || mutation.isPending}
                  loading={mutation.isPending}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <Button
          size="icon"
          onClick={() => setOpen((v) => !v)}
          className="h-12 w-12 rounded-full shadow-popover"
        >
          {open ? <X className="h-5 w-5" /> : <MessageCircle className="h-5 w-5" />}
        </Button>
      </div>
    </>
  )
}
