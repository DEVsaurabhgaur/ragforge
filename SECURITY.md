# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes     |
| 1.x     | ❌ No      |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.
Instead, email the maintainer directly: contact via GitHub profile.

We will respond within 48 hours and coordinate a fix before public disclosure.

## Best Practices

- **Never commit your `.env` file** — it contains API keys.
- Use environment variables or secrets managers in production.
- Regularly rotate your Gemini/OpenAI API keys.
- The `.gitignore` already excludes `.env` by default.
