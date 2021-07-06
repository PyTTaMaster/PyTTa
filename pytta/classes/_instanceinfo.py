# -*- coding: utf-8 -*-
"""
    INSTANCE INFO:
    -----------------

        This module is a copy of the code created by TheoRet at StackOverflow

    TheoRet profile:
        https://stackoverflow.com/users/7386061/theoret

    Link to the question/answer:
        https://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init/49331683#49331683
"""


import time
import traceback
import threading


class InstanceCreationError(Exception):
    pass


class RememberInstanceCreationInfo:
    def __init__(self):
        for frame, _ in traceback.walk_stack(None):
            varnames = frame.f_code.co_varnames
            if varnames == ():
                break
            if frame.f_locals[varnames[0]] not in (self, self.__class__):
                break
                # if the frame is inside a method of this instance,
                # the first argument usually contains either the instance or
                #  its class
                # we want to find the first frame, where this is not the case
        else:
            raise InstanceCreationError("No suitable outer frame found.")
        self._outer_frame = frame
        self.creation_module = frame.f_globals["__name__"]
        self.creation_file, self.creation_line, self.creation_function, \
            self.creation_text = \
            traceback.extract_stack(frame, 1)[0]
        self.creation_name = self.creation_text.split("=")[0].strip()
        super().__init__()
        # threading.Thread(target=self._check_existence_after_creation).start()

    def _check_existence_after_creation(self):
        while self._outer_frame.f_lineno == self.creation_line:
            time.sleep(0.01)
        # this is executed as soon as the line number changes
        # now we can be sure the instance was actually created
        error = InstanceCreationError(
                "\nCreation name not found in creation frame.\ncreation_file: "
                "%s \ncreation_line: %s \ncreation_text: %s\ncreation_name ("
                "might be wrong): %s" % (
                    self.creation_file, self.creation_line, self.creation_text,
                    self.creation_name))
        nameparts = self.creation_name.split(".")
        try:
            var = self._outer_frame.f_locals[nameparts[0]]
        except KeyError:
            raise error
        finally:
            del self._outer_frame
        # make sure we have no permanent inter frame reference
        # which could hinder garbage collection
        try:
            for name in nameparts[1:]:
                var = getattr(var, name)
        except AttributeError:
            raise error
        if var is not self:
            raise error

    def __repr__(self):
        return super().__repr__()[
               :-1] + " with creation_name '%s'>" % self.creation_name
