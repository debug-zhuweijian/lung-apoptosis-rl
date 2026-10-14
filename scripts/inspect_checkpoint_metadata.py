"""Bounded protocol-2 symbolic inspection. Never reconstruct pickle objects.

GLOBAL names, REDUCE/BUILD and persistent references remain inert symbols.
Only explicit primitive metadata fields are exported; tensor contents are unread.
"""
import argparse
import hashlib
import json
import math
import pickletools
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024


@dataclass
class Symbol:
    kind: str
    args: object = None
    items: dict = field(default_factory=dict)


def parse_symbols(data):
    """Interpret container opcodes only, never dispatch named constructors."""
    stack, memo = [], {}
    mark = object()
    protocol = False
    names = set()

    def take(n=1):
        if len(stack) < n or any(value is mark for value in stack[-n:]):
            raise ValueError("operand underflow or MARK used as a value")
        values = stack[-n:]
        del stack[-n:]
        return values

    def top():
        if not stack or stack[-1] is mark:
            raise ValueError("operand underflow or MARK used as a value")
        return stack[-1]

    def marked():
        if not any(value is mark for value in stack):
            raise ValueError("missing MARK")
        index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is mark)
        values = stack[index + 1:]
        del stack[index:]
        return values

    def put_items(target, values):
        if len(values) % 2:
            raise ValueError("odd dictionary items")
        dest = target if type(target) is dict else target.items if isinstance(target, Symbol) else None
        if dest is None:
            raise ValueError("invalid SETITEM target")
        for key, value in zip(values[::2], values[1::2]):
            if type(key) not in (str, int):
                raise ValueError("nonprimitive dictionary key")
            if key in dest:
                raise ValueError("duplicate dictionary key")
            dest[key] = value

    for count, (op, arg, pos) in enumerate(pickletools.genops(data)):
        name = op.name
        names.add(name)
        if count > 1000000:
            raise ValueError("opcode limit")
        if count == 0 and (name != "PROTO" or arg != 2):
            raise ValueError("only explicit protocol 2 supported")
        if name == "PROTO":
            if protocol or arg != 2:
                raise ValueError("invalid protocol")
            protocol = True
        elif name == "MARK":
            stack.append(mark)
        elif name in ("BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT"):
            stack.append(arg)
        elif name in ("NONE", "NEWTRUE", "NEWFALSE"):
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "TUPLE":
            stack.append(tuple(marked()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            stack.append(tuple(take(n)))
        elif name in ("BINPUT", "LONG_BINPUT"):
            if arg in memo:
                raise ValueError("duplicate memo index")
            memo[arg] = top()
        elif name in ("BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif name == "GLOBAL":
            stack.append(Symbol("global", arg))
        elif name in ("REDUCE", "NEWOBJ"):
            constructor, arguments = take(2)
            if type(arguments) is not tuple:
                raise ValueError("constructor arguments must be a tuple")
            stack.append(Symbol(name, (constructor, arguments)))
        elif name == "BINPERSID":
            stack.append(Symbol("persistent", take()[0]))
        elif name == "BUILD":
            take()  # state is deliberately never applied
            if not isinstance(top(), Symbol):
                raise ValueError("BUILD target must be opaque")
        elif name == "SETITEM":
            key, value = take(2)
            put_items(top(), [key, value])
        elif name == "SETITEMS":
            values = marked()
            put_items(top(), values)
        elif name == "APPEND":
            value = take()[0]
            if type(top()) is not list:
                raise ValueError("invalid APPEND")
            top().append(value)
        elif name == "APPENDS":
            values = marked()
            if type(top()) is not list:
                raise ValueError("invalid APPENDS")
            top().extend(values)
        elif name == "STOP":
            if len(stack) != 1 or type(stack[0]) is not dict or pos != len(data) - 1:
                raise ValueError("invalid root or trailing bytes")
            return stack[0], sorted(names)
        else:
            raise ValueError("unsupported opcode: " + name)
    raise ValueError("missing STOP")


def scalar(value):
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return "unknown (non-scalar or opaque)"


def inspect(path, max_bytes=MAX_BYTES):
    path = Path(path)
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        members = [i for i in archive.infolist() if i.filename.endswith("/data.pkl") or i.filename == "data.pkl"]
        if len(members) != 1:
            raise ValueError("exactly one data.pkl required")
        entry = members[0]
        if entry.file_size > max_bytes:
            raise ValueError("data.pkl exceeds byte cap")
        try:
            with archive.open(entry) as stream:
                data = stream.read(max_bytes + 1)
        except (NotImplementedError, RuntimeError, zlib.error) as error:
            raise ValueError("unsupported or unreadable ZIP payload") from error
        if len(data) > max_bytes:
            raise ValueError("data.pkl exceeds byte cap")
    root, opcodes = parse_symbols(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed during inspection")
    fields = {k: scalar(root[k]) for k in ("schema_version", "epoch", "identity") if k in root}
    for container, keys in (("metadata", ("stage", "data_identity", "synthetic")), ("counters", ("optimizer_step",))):
        values = root.get(container)
        if type(values) is dict:
            fields[container] = {k: scalar(values[k]) for k in keys if k in values}
        elif container in root:
            fields[container] = "unknown (opaque or non-dictionary)"
    return {"path": str(path.resolve()), "file_size": before.st_size,
            "mtime_ns": before.st_mtime_ns, "data_pkl_size": len(data),
            "data_pkl_sha256": hashlib.sha256(data).hexdigest(), "opcodes": opcodes,
            "fields": fields, "model_tensor_shapes": "unknown (constructors remain opaque)",
            "certifies": "static serialized metadata only; no tensor finiteness, model behavior or joint compatibility"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.paths) > 4:
        parser.error("at most four explicit paths")
    results = []
    for path in args.paths:
        try:
            results.append({"status": "inspected", **inspect(path)})
        except (ValueError, KeyError, IndexError, OSError, zipfile.BadZipFile) as error:
            results.append({"status": "rejected", "path": path, "reason": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{"path": r["path"], "status": r["status"]} for r in results], ensure_ascii=False))


if __name__ == "__main__":
    main()
