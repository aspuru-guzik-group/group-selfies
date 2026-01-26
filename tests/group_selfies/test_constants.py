

import group_selfies.constants

def test_index_length():
    assert len(group_selfies.constants.INDEX_ALPHABET) == 16, "INDEX ALPHABET not in hexadecimal"
    