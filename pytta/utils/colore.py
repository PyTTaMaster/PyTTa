# -*- coding: utf-8 -*-
"""
Módulo simples para colorir textos

@author: João Vitor G. Paes
"""



__STRCLR = {
    "none": "",
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "pink": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "clear": "\x1b[0m"
}


__BGRCLR = {
    "none": "",
    "black": "\x1b[40m",
    "red": "\x1b[41m",
    "green": "\x1b[42m",
    "yellow": "\x1b[43m",
    "blue": "\x1b[44m",
    "pink": "\x1b[45m",
    "cyan": "\x1b[46m",
    "white": "\x1b[47m",
    "clear": "\x1b[0m"
}


_LINEOBJ = {
    "clear": "0",
    "font": "3",
    "background": "4",
}


_COLORS = {
    "clear": "",
    "black": "0",
    "red": "1",
    "green": "2",
    "yellow": "3",
    "blue": "4",
    "pink": "5",
    "cyan": "6",
    "white": "7"
}


class Error(Exception):
    pass


class ColorNameError(Error):
    def __init__(self):
        """
        Error raised if color not implemented.

        Returns:
            None.

        """
        self.message = "This color is not available."
        return


class ColorStr(object):
    """Color string"""
    
    def __init__(self, font: str = "clear", back: str = "clear") -> None:
        """
        Pintor de linhas.

            Example use:

                >>> black_on_white_str = ColorStr("black", "white")
                >>> red_on_green_str = ColorStr("red", "green")
                >>> print(black_on_white_str("Estou usando colore.ColorStr"))
                >>> print(red_on_green_str("I'm using colore.ColorStr"))

        Args:
            font (str, optional): Cor da fonte | Font color. Defaults to "clear".
            back (str, optional): Cor do fundo | Background color. Defaults to "clear".

        Returns:
            None.

        """
        self.fntclr = font
        self.bgrclr = back
        return

    def __call__(self, text: str = None) -> str:
        """
        Paint the text with its font and background colors.

        Args:
            text (str, optional): The text to be painted. Defaults to None.

        Returns:
            str: Colored text and background.

        """
        if text is None:
            text = ColorStr.__str__(self)
        return colorir(text, self.fntclr, self.bgrclr)

    @property
    def fntclr(self) -> str:
        """Font color."""
        return self._fntclr

    @fntclr.setter
    def fntclr(self, color: str) -> None or Error:
        if color in _COLORS.keys():
            self._fntclr = color
        else:
            raise ColorNameError

    @property
    def font(self) -> str:
        """Alias for fntclr."""
        return self.fntclr

    @property
    def bgrclr(self) -> str:
        """Background color."""
        return self._bgrclr

    @bgrclr.setter
    def bgrclr(self, color: str) -> None or Error:
        if color in _COLORS.keys():
            self._bgrclr = color
        else:
            raise ColorNameError

    @property
    def background(self) -> str:
        """Alias for bgrclr."""
        return self.bgrclr

    @property
    def back(self) -> str:
        """Alias for bgrclr."""
        return self.bgrclr


def _color_code(which: str, color: str) -> str:
    return f'\x1b[{_LINEOBJ[which]}{_COLORS[color]}m'


def _color_clear() -> str:
    return _color_code("clear", "clear")


def _pintor(text: str, which: str, color: str) -> str or Error:
    try:
        return f'{_color_code(which, color)}{text}{_color_clear()}'
    except KeyError:
        raise ColorNameError


def pinta_texto(text: str, color: str) -> str:
    """
    Pinta o texto com a cor escolhida.

    Args:
        text (str): O texto em si.
        color (str): A cor escolhida.

    Returns:
        str: Texto colorido.

    """
    return _pintor(text, "font", color)


def pinta_fundo(text: str, color: str) -> str:
    """
    Pinta o fundo do texto com a cor escolhida.

    Args:
        text (str): O texto em si.
        color (str): A cor escolhida.

    Returns:
        str: Texto com fundo colorido.

    """
    return _pintor(text, "background", color)


def colorir(texto: str, fntclr: str = None, bgrclr: str = None) -> str:
    """
    Retorna o texto colorido nas cores escolhidas.

    Caso nenhuma cor seja informada, retorna o texto sem alterações.

    Args:
        texto (str): DESCRIPTION.
        fntclr (str, optional): Font color. Defaults to None.
        bgrclr (str, optional): Background color. Defaults to None.

    Returns:
        texto (str): Colored font and background.

    """
    if fntclr is None:
        fntclr = "clear"
    elif bgrclr is None:
        bgrclr = "clear"
    texto = pinta_texto(texto, fntclr)
    texto = pinta_fundo(texto, bgrclr)
    return texto


