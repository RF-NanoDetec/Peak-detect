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

    console.log(`[WebSocket] Connecting for task ${taskId}`)

    // Connect to WebSocket
    wsClient.connect(taskId)
    setIsConnected(true)
    lastMessageTimeRef.current = Date.now()

    // Subscribe to messages
    unsubscribeRef.current = wsClient.subscribe(taskId, (msg: WSProgressMessage) => {
      console.log(`[WebSocket] Received message:`, msg)
      setMessage(msg)
      lastMessageTimeRef.current = Date.now()
    })

    // Set up polling fallback after 3 seconds of no WebSocket messages
    wsTimeoutRef.current = setTimeout(() => {
      console.log(`[WebSocket] No message received in 3s, starting polling fallback`)
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

    console.log(`[Polling] Starting fallback polling for task ${taskId}`)
    
    const poll = async () => {
      try {
        const status = await apiClient.getTaskStatus(taskId)
        console.log(`[Polling] Task status:`, status)
        
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
          console.log(`[Polling] Task ${status.status}, stopping polling`)
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

