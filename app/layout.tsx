import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SmartMine — EfficientNetB0 Safety Classifier",
  description: "AI-powered mining safety image classifier using EfficientNetB0",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
