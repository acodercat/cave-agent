## Contribute to CaveAgent

We welcome contributions to CaveAgent! Here's how you can help:

### Types of Contributions

- Bug fixes
- Feature additions
- Documentation improvements
- Test coverage improvements

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install the development environment (`uv sync --all-groups`)
4. Make your changes
5. Run the quality gates:

   ```bash
   ruff check .
   ruff format --check .
   pyright
   pytest
   python -m build
   ```

   The first four are what CI runs on 3.12 and 3.13; a green local run is a
   green PR. `pytest` skips the tests that need a live LLM — set
   `LLM_MODEL_ID`, `LLM_API_KEY` and `LLM_BASE_URL` to include them, and note
   that they spend tokens against whatever endpoint you point them at.

6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request
