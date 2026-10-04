import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from typing_extensions import override

from capymoa.core import LabeledInstance, RegressionInstance, _AnyInstance

from ._stream import FiniteStream, Schema


class CSVStream(FiniteStream[_AnyInstance]):
    """Create a CapyMOA datastream from a CSV file.

    * The CSV file **must** have a header row with feature names.
    * Integers or strings can specify nominal features.
    * ``?`` represent missing values.
    * CSV is read line by line, so it can handle large files.

    When 'categories' are provided for the target attribute, then the stream returns
    :class:`~capymoa.core.LabeledInstance` objects.

    >>> from io import StringIO
    >>> from capymoa.stream import CSVStream
    >>> csv_content = '''feature1,feature2,target
    ... 1,A,yes
    ... 2,B,no
    ... 3,0,0
    ... 5,1,1
    ... ?,?,?
    ... '''
    >>> csv_file = StringIO(csv_content)
    >>> stream = CSVStream(
    ...     file=csv_file,
    ...     target="target",
    ...     categories={"target": ["yes", "no"], "feature2": ["A", "B"]},
    ...     name="TestStream"
    ... )
    >>> for instance in stream:
    ...     print(instance.x, instance.y_index, instance.y_label)
    [1. 0.] 0 yes
    [2. 1.] 1 no
    [3. 0.] 0 yes
    [5. 1.] 1 no
    [nan nan] -1 None

    When no categories are provided for the target attribute, then the stream returns
    :class:`~capymoa.core.RegressionInstance` objects.

    >>> csv_content = '''target,feature1,feature2
    ... 0.0,A,1
    ... 0.5,B,2
    ... 1.5,0,3
    ... 2.0,1,4
    ... ?,?,?
    ... '''
    >>> csv_file = StringIO(csv_content)
    >>> stream = CSVStream(
    ...     file=csv_file,
    ...     target="target",
    ...     categories={"feature1": ["A", "B"]},
    ...     name="TestStream"
    ... )
    >>> for instance in stream:
    ...     print(instance.x, instance.y_value)
    [0. 1.] 0.0
    [1. 2.] 0.5
    [0. 3.] 1.5
    [1. 4.] 2.0
    [nan nan] nan

    """

    def __init__(
        self,
        file: Path | str | TextIO,
        target: str,
        categories: Mapping[str, Sequence[str]] | None = None,
        name: str | None = None,
        start: float = 0,
        end: float | None = None,
    ) -> None:
        """Create a CSV stream.

        Reads the file once to count the rows.

        :param file: A path to a CSV file or an open file-like object.
        :param target: The name of the target attribute.
        :param categories: A mapping from attribute names to their categorical values.
        :param name: An optional name for the stream. If not provided, the filename is
            used.
        :param start: Index of the first instance to read. See :class:`FiniteStream`.
        :param end: Index to stop reading at (excluded). See :class:`FiniteStream`.
        """

        # Open file if a path is given
        if isinstance(file, (str, Path)):
            file = Path(file).expanduser()
            name = name or file.stem
            # Kept open for the lifetime of the stream (supports `restart()`
            # via seek), so it can't be scoped to a `with` block here.
            self._file = open(file, "r")  # noqa: SIM115
        else:
            self._file = file

        if categories and target in categories:
            self._instance_type = LabeledInstance
        else:
            self._instance_type = RegressionInstance

        self._reader = csv.reader(self._file, delimiter=",")
        feature_names = next(self._reader)
        feature_names = [n.strip() for n in feature_names]

        if target not in feature_names:
            raise ValueError(
                f"Target attribute '{target}' not found in CSV header. "
                f"Header contains {feature_names}."
            )

        self.schema = Schema.from_custom(
            features=feature_names,
            target=target,
            categories=categories,
            name=name or "unnamed-csv-stream",
        )
        super().__init__(sum(1 for _ in self._reader), start, end)

    def _read_instance(self) -> _AnyInstance | None:
        try:
            return self._instance_type.from_csv_row(self.schema, next(self._reader))  # type: ignore
        except StopIteration:
            return None

    @override
    def get_schema(self) -> Schema:
        return self.schema

    @override
    def _has_more_instances(self) -> bool:
        return self._instance is not None

    @override
    def _next_instance(self) -> _AnyInstance:
        instance = self._instance
        if instance is None:
            raise StopIteration()
        self._instance = self._read_instance()
        return instance

    @override
    def _seek(self, index: int) -> None:
        if self._file.seek(0) != 0:
            raise OSError("Failed to seek to the beginning of the file.")
        self._reader = csv.reader(self._file, delimiter=",")
        next(self._reader)  # Skip header
        for _ in range(index):
            next(self._reader)
        # Buffer a single instance so that we can implement has_more_instances
        self._instance = self._read_instance()
