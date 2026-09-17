'use server';

// This is a server action. You can use it like an API endpoint on the server but it can be accessed directly inside a component.
// This lets you do things like use secrets that are only on the server, access the filesystem, or query a DB
// See where it's used in the Form component for an example
// Check out the NextJS docs if you'd like to learn more about them: https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions-and-mutations
export async function solve(data: { question: string; choices: Array<string> }) {
    const origin = process.env.API_ORIGIN ?? 'http://api:8000';
    const resp = await fetch(`${origin}/api/solve`, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json' },
    });
    if (!resp.ok) {
        throw new Error(`API request failed: ${resp.status}: ${resp.statusText}`);
    }

    console.log('This log will show up in the NextJS server logs!');
    return await resp.json();
}
