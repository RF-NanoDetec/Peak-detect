"use client"

import { useEffect, useRef, useState } from 'react'
import { wsClient, apiClient } from '@/lib/apiClient'
import type { WSProgressMessage } from '@/lib/types'

export function useWebSocket(taskId: string | null) {
  const [message, setMessage] = useState<WSProgressMessage | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const unsubscribeRef = useRef<(() => void) | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const wsTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const lastMessageTimeRef = useRef<number>(Date.now())

  useEffect(() => {
    if (!taskId) {
      setIsConnected(false)
      return
    }

    // Connect to WebSocket
    wsClient.connect(taskId)
    setIsConnected(true)
    lastMessageTimeRef.current = Date.now()

    // Subscribe to messages
    unsubscribeRef.current = wsClient.subscribe(taskId, (msg: WSProgressMessage) => {
      setMessage(msg)
      lastMessageTimeRef.current = Date.now()
      if (wsTimeoutRef.current) {
        clearTimeout(wsTimeoutRef.current)
        wsTimeoutRef.current = null
      }
    })

    // Set up polling fallback after 3 seconds of no WebSocket messages
    wsTimeoutRef.current = setTimeout(() => {
      startPolling(taskId)
    }, 3000)

    // Cleanup on unmount or taskId change
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
        unsubscribeRef.current = null
      }
      if (wsTimeoutRef.current) {
        clearTimeout(wsTimeoutRef.current)
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      wsClient.disconnect()
      setIsConnected(false)
    }
  }, [taskId])

  const startPolling = (taskId: string) => {
    if (pollingIntervalRef.current) return // Already polling
    
    const poll = async () => {
      try {
        const status = await apiClient.getTaskStatus(taskId)
        
        const msg: WSProgressMessage = {
          type: 'progress',
          taskId,
          progress: status.progress || 0,
          message: '',
          status: status.status as any,
          result: status.result,
          error: status.error,
        }
        
        setMessage(msg)
        
        // Stop polling if task is complete or failed
        if (status.status === 'completed' || status.status === 'failed') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
        }
      } catch (error) {
        console.error(`[Polling] Error polling task status:`, error)
      }
    }

    // Poll immediately then every second
    poll()
    pollingIntervalRef.current = setInterval(poll, 1000)
  }

  return {
    message,
    isConnected,
    progress: Math.round(message?.progress || 0),
    status: message?.status || 'pending',
    result: message?.result,
    error: message?.error,
  }
}
