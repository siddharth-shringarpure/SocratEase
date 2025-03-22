"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import Link from "next/link";

export default function LandingPage() {
  const sections = [
    {
      title: "Practice Communication",
      description: "Practice communication with real-time AI feedback",
      href: "/practice",
      icon: "👥",
      color: "from-blue-500/20 to-cyan-500/20",
    },
    {
      title: "Analytics",
      description: "Track your communication progress",
      href: "/analytics",
      icon: "📈",
      color: "from-violet-500/20 to-purple-500/20",
    },
  ];

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl mx-auto">
        {/* Hero */}
        <div className="text-center mb-20">
          <div className="inline-block mb-6 text-4xl">🎯</div>
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            Socrat<span className="text-primary">Ease</span>
          </h1>
          <p className="text-xl text-muted-foreground">
            Master the art of communication
          </p>
        </div>

        {/* Navigation Cards */}
        <div className="space-y-4">
          {sections.map((section) => (
            <Card
              key={section.title}
              className="group relative border-2 overflow-hidden"
            >
              <div
                className={`absolute inset-0 bg-gradient-to-r ${section.color} opacity-0 group-hover:opacity-100 transition-opacity`}
              />
              <Link href={section.href} className="block">
                <CardHeader className="py-5 relative">
                  <div className="flex items-center gap-4">
                    <span className="text-3xl">{section.icon}</span>
                    <div className="flex-1">
                      <CardTitle className="text-xl mb-1">
                        {section.title}
                      </CardTitle>
                      <CardDescription className="text-sm">
                        {section.description}
                      </CardDescription>
                    </div>
                    <span className="text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                      →
                    </span>
                  </div>
                </CardHeader>
              </Link>
            </Card>
          ))}
        </div>

        <footer className="text-center text-sm text-muted-foreground/60 mt-20">
          <p>Empowering better communication through AI 🚀</p>
        </footer>
      </div>
    </main>
  );
}
