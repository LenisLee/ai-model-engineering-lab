from typing import Callable, Dict, TypeVar, Generic, Type

T = TypeVar("T")


class Registry(Generic[T]):
    """A generic registry for registering and retrieving named components."""

    def __init__(self, name: str = "registry"):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T = None) -> Callable[[T], T]:
        if item is not None:
            self._items[name] = item
            return item

        def decorator(item: T) -> T:
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"{name!r} not found in {self.name} registry")
        return self._items[name]

    def list(self) -> list[str]:
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


# Top-level registries
model_registry = Registry[Type]("model")
dataset_registry = Registry[Type]("dataset")
trainer_registry = Registry[Type]("trainer")
pipeline_registry = Registry[Type]("pipeline")
