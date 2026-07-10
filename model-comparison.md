# Model Comparison — "Explain API"

## Prompt Used
"Explain API"

## Responses

### llama3
- Detailed explanation with steps (Request, Gateway, Endpoint, Response)
- Covered types: REST, SOAP, GraphQL
- Covered industries: E-commerce, Healthcare, Finance
- Speed: This one took noticeably longer than the other two — I was waiting a few extra seconds before the response started coming through
- Length: By far the longest response of the three — it went into the full request lifecycle and named all three protocol types
- Quality: The most thorough of the three. If I wanted to actually learn how APIs work rather than just get a definition, this was the most useful

### mistral
- Clear and practical explanation
- Gave real world examples: Stripe, Twitter, Google Maps
- Speed: Slower than phi3 but faster than llama3 — somewhere in the middle
- Length: Shorter than llama3 since it skipped the protocol-level detail and went straight to examples instead
- Quality: This was the only response that named actual companies (Stripe, Twitter, Google Maps), which made it easier to picture what an API actually does in practice rather than just reading a definition

### phi3
- Clean concise summary
- Covered public, private and partner APIs
- Speed: The fastest of the three — response came back almost immediately
- Length: Shortest by far — just a definition plus one way of categorizing APIs (by access level)
- Quality: Not as deep as the other two, but if I just needed a quick refresher this was the fastest way to get one

## Comparison Table

| | llama3 | mistral | phi3 |
|---|---|---|---|
| Speed | Slowest — took noticeably longer | Moderate | Fastest — near instant |
| Length | Longest | Medium | Shortest |
| What it covered | Protocol types (REST/SOAP/GraphQL) + industries | Real-world examples (Stripe, Twitter, Google Maps) | Access-level types (Public/Private/Partner) |
| Best For | Teaching/docs — I'd use this if I actually needed to understand the mechanics | General use — good balance of clarity and real examples | Quick reference — fastest when I just need a reminder |