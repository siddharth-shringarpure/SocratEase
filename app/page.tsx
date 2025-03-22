import Link from "next/link";

export default async function Home() {
  const response = await fetch("http://localhost:5328/api/python");
  const text = await response.text();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold mb-4">Python API Response</h1>
      <p className="text-lg">{text}</p>
    </main>
  );
}
