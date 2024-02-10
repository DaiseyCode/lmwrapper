from lmwrapper.utils import flatten_dict


def test_dict_flatten():
    result = flatten_dict({"a": "foo", "b": {"c": "bar", "d": {"e": "baz"}}})
    expected = {"a": "foo", "b__c": "bar", "b__d__e": "baz"}
    assert result == expected
