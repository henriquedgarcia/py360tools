from typing import Callable


class LazyProperty:
    _setter: Callable = None
    _deleter: Callable = None
    name: str
    attr_name: str

    def __init__(self, getter):
        """
        Creates a property that waits for the first use to be initialized. After this, it always returns the same
        result.

        Usage:

        class Bar:
            @LazyProperty
            def foo(self):
                print('calculating... ', end='')
                value = 1+1
                return value

            @foo.setter
            def foo(self, value):
                print('Setting foo with value automatically.')

        test = Bar()
        print(test.foo)  # print 'calculating... 2'
        print(test.foo)  # print '2'
        test.foo = 4     # print 'Setting foo with value automatically.'
        print(test.foo)  # print '4'. No calculate anymore.

        The value is stored in Bar._foo only once.

        :param getter:
        :type getter: Callable
        """
        self.getter = getter
        self.name = self.getter.__name__
        self.attr_name = '_' + self.name

    def __get__(self, instance, owner):
        """
        Run getter after getattr

        :param instance:
        :param owner:
        :return:
        """
        try:
            value = getattr(instance, self.attr_name)
        except AttributeError:
            value = self.getter(instance)
            setattr(instance, self.attr_name, value)
        return value

    def __set__(self, instance, value):
        """
        Run setter after setattr

        :param instance:
        :param value:
        :return:
        """
        setattr(instance, self.attr_name, value)

        if self._setter is not None:
            self._setter(instance, value)

    def __delete__(self, instance):
        """
        Run deleter before delattr
        :param instance:
        :return:
        """
        if self._deleter is not None:
            self._deleter(instance)

        delattr(instance, '_' + self.name)

    def setter(self, value: Callable):
        self._setter = value

    def deleter(self, value: Callable):
        self._deleter = value
