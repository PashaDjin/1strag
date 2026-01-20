# Anti-Overengineering Rules

## Core Principles
1. **One file = One task** - Keep each module focused on a single responsibility
2. **No abstraction layers** - Write direct, simple code without unnecessary wrappers
3. **Minimal dependencies** - Use only what's strictly necessary
4. **No premature optimization** - Make it work first, optimize only if needed
5. **Direct implementation** - Avoid design patterns unless absolutely required
6. **Keep it simple** - If you can delete code, do it

## File Structure
- `app.py` - Streamlit UI only
- `rag_setup.py` - RAG configuration and setup only
- `requirements.txt` - Dependencies only
- `books/` - PDF storage only

## Code Guidelines
- Use explicit imports
- No custom classes unless necessary
- Prefer functions over objects
- Keep functions short and focused
- No config files - hardcode sensible defaults
- Comments only when necessary
