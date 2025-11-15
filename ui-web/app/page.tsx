"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function Home() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to load page on initial visit
    router.push('/load')
  }, [router])

  return null
}
