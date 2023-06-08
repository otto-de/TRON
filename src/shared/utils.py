def get_offsets(sessions_path):
        line_offsets = []
        with open(sessions_path, "rt") as f:
            offset = 0
            for line_idx, line in enumerate(f):
                line_len = len(line)
                line_offsets.append((line_len, line_idx, offset))
                offset += line_len
        line_offsets = [offset for _, _, offset in line_offsets]
        return line_offsets

