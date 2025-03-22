import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AnalyticsPage() {
  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">
        Communication Analytics
      </h1>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 max-w-5xl w-full">
        <Card>
          <CardHeader>
            <CardTitle className="text-center">Practice Sessions</CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">12</div>
            <p className="text-muted-foreground">Total sessions this month</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-center">Average Score</CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">85%</div>
            <p className="text-muted-foreground">Across all sessions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-center">Time Practiced</CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">5.2h</div>
            <p className="text-muted-foreground">Total practice time</p>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
