## Summary

- What changed?
- Why is it needed?

## Validation

- [ ] `uv run pytest`
- [ ] `uv run python -m compileall -q src tests`
- [ ] `uv build`
- [ ] Manual digiKam workflow checked when applicable

## Safety

- [ ] No media, models, credentials, databases, or generated proxies are included
- [ ] digiKam database access remains read-only
- [ ] Cleanup targets only manifest-owned generated files
