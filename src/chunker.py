"""Splits corpus into chunks."""

# define start and end of a piece as a tuple
Range = tuple[int, int]


def split_python(text: str) -> list[Range]:
    """Cut Python code before each def/class, keeping decorators attached"""
    pieces: list[Range] = []
    piece_start = 0
    position = 0
    after_decorator = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        starts = stripped.startswith(("def ", "async def ", "class ", "@"))
        if starts and not after_decorator and position > piece_start:
            pieces.append((piece_start, position))
            piece_start = position
        after_decorator = stripped.startswith("@")
        position += len(line)
    if position > piece_start:
        pieces.append((piece_start, position))
    return pieces


def split_markdown(text: str) -> list[Range]:
    """Cut Markdown before each heading, ignoring '#' inside code blocks."""
    pieces: list[Range] = []
    piece_start = 0
    position = 0
    in_code_block = False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            in_code_block = not in_code_block
        is_heading = line.startswith("#") and not in_code_block
        if is_heading and position > piece_start:
            pieces.append((piece_start, position))
            piece_start = position
        position += len(line)
    if position > piece_start:
        pieces.append((piece_start, position))
    return pieces


def cut_long_piece(text: str, start: int, end: int,
                   max_size: int) -> list[Range]:
    """Cut a piece longer than max_size on line ends, or mid-line if forced."""
    spans: list[Range] = []
    while end - start > max_size:
        # Prefer to cut just after the last newline that still fits.
        cut = text.rfind("\n", start, start + max_size) + 1
        if cut <= start:
            cut = start + max_size
        spans.append((start, cut))
        start = cut
    spans.append((start, end))
    return spans


def pack(text: str, pieces: list[Range], max_size: int) -> list[Range]:
    """Cut long pieces, then append consecutive ones while they fit."""
    chunks: list[Range] = []
    for start, end in pieces:
        for cut_start, cut_end in cut_long_piece(text, start, end, max_size):
            if chunks and cut_end - chunks[-1][0] <= max_size:
                chunks[-1] = (chunks[-1][0], cut_end)
            else:
                chunks.append((cut_start, cut_end))
    return chunks


def chunk_text(text: str, is_python: bool, max_size: int) -> list[Range]:
    """Chunk a whole file and drop chunks that are only whitespace."""
    if is_python:
        pieces = split_python(text)
    else:
        pieces = split_markdown(text)
    chunks = []
    for start, end in pack(text, pieces, max_size):
        if text[start:end].strip():
            chunks.append((start, end))
    return chunks
