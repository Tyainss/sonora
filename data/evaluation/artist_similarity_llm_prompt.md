# Independent Musical Similarity Evaluation - Sonora

You are acting as a music expert with extensive knowledge of artists, genres, musical styles, discographies and artist relationships.

I am building **Sonora**, a music intelligence project that identifies musically similar artists using machine learning.

I want you to create an **independent, high-quality reference dataset** that will help evaluate different artist similarity approaches.

## Your task

I will provide a dataset containing my **500 most-listened artists**.

For each of the 15 reference artists below, identify and rank the **5 most musically similar artists from that exact list of 500**.

The reference artists are:

1. Silvana Estrada
2. IDLES
3. billy woods
4. JPEGMAFIA
5. Explosions in the Sky
6. Fleet Foxes
7. Kanye West
8. Geese
9. Linda Martini
10. Big Thief
11. Little Simz
12. Air
13. Sabrina Carpenter
14. Ichiko Aoba
15. Portishead

## How to evaluate similarity

Your primary objective is to identify artists whose **music genuinely resembles that of the reference artist**.

Consider characteristics such as:

- Genres and subgenres, including specific stylistic differences within broad genres.
- Sound, production style, instrumentation and arrangements.
- Vocal characteristics and musical delivery.
- Songwriting, composition and musical experimentation.
- Overall musical atmosphere and aesthetic.
- Musical influences and artistic relationships, where relevant.

**Important considerations:**

- Artists sharing band members, solo projects or related musical projects are valid and potentially very strong matches. Consider these relationships explicitly, but don't assume they guarantee identical musical styles.
- Don't prioritize artists merely because they're popular, from the same country, active in the same period or share a broad genre.
- Avoid selecting artists solely because they have collaborated or influenced each other. The actual music should remain the primary consideration.
- Consider the artist's broader discography rather than judging similarity from one particular song or album.
- Take particular care with less internationally known artists, including Portuguese, Brazilian and independent musicians.

## Research and independence

**I want a genuinely independent evaluation, not a reconstruction of another similarity system.**

1. Review the complete list of 500 artists before making your selections. Don't simply select the first familiar names.
2. Use your existing musical knowledge and conduct additional research where necessary, especially for unfamiliar artists.
3. When researching, prioritize descriptions of musical characteristics, albums, instrumentation and production. Don't copy recommendations or similarity rankings from Last.fm, Spotify or other recommendation systems.
4. Do not rely on any existing Sonora predictions, previous evaluations or user preferences.
5. Never invent artist information. If you cannot confidently evaluate an obscure artist, acknowledge that limitation.
6. Be willing to select unconventional or less obvious matches when there is strong musical justification.

I value **accuracy and thoughtful musical judgment over speed**. Investigate questionable candidates rather than defaulting to familiar names.

## Candidate restrictions

- Every suggested artist **must exist in the attached top-500 dataset**.
- Use the exact canonical artist name from the dataset.
- Include the canonical artist ID if provided.
- Never recommend the reference artist itself.
- Do not repeat a candidate within the same top 5.
- An artist can appear in multiple reference lists.

## Output

For each reference artist, provide a table containing:

| Rank | Similar artist | Why they are similar                  |
| ---- | -------------- | ------------------------------------- |
| 1    | Artist         | Brief, specific musical justification |
| 2    | Artist         | Brief, specific musical justification |
| 3    | Artist         | Brief, specific musical justification |
| 4    | Artist         | Brief, specific musical justification |
| 5    | Artist         | Brief, specific musical justification |

Rank artists from **1 (strongest match) to 5**.

Keep explanations concise but meaningful. Avoid generic statements such as "both are alternative artists" when more specific musical similarities can be identified.

After the rankings, briefly highlight any selections where your confidence is limited or where another candidate was particularly close.

**Do not ask me to review or guide your selections while working.** Complete the independent evaluation first.

The final goal is to produce a reproducible reference dataset that we can compare with Sonora's machine learning results. Your selections will be treated as informed judgments, not unquestionable ground truth.