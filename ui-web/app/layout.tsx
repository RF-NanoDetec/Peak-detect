import type { Metadata } from "next";
import { Inter, Roboto_Mono, Rajdhani } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { Toaster } from "sonner";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const robotoMono = Roboto_Mono({ subsets: ["latin"], variable: "--font-mono" });
const rajdhani = Rajdhani({ 
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-display"
});

export const metadata: Metadata = {
  title: "Peak Analysis Tool",
  description: "LabOne-style peak analysis application",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${robotoMono.variable} ${rajdhani.variable} font-sans tracking-tight`}>
        <div className="h-screen flex flex-col">
          <Topbar />
          <div className="flex-1 flex overflow-hidden min-h-0">
            <Sidebar />
            <main className="flex-1 flex flex-col overflow-hidden min-h-0">
              {children}
            </main>
          </div>
        </div>
        <Toaster />
      </body>
    </html>
  );
}
