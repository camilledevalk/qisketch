# Import necessary libraries
from qiskit.circuit.library import XGate, YGate, ZGate, HGate, PhaseGate, TGate, SGate, SdgGate, IGate
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
from qiskit import QuantumCircuit
import matplotlib.colors as mpl_colors
import pylatexenc


# Define available quantum gates
GATES = [XGate, YGate, ZGate, HGate, TGate, SGate, SdgGate, IGate]

# Define corresponding gate names
GATE_NAMES = ["x", "y", "z", "h", "t", "s", "sdg", "i"]

# Define gates that cannot be used as control gates
FORBIDDEN_CONTROL_GATES = [ZGate, PhaseGate]

# Create a list of control gates
CONTROL_GATES = list(GATES)
CONTROL_GATE_NAMES = list(GATE_NAMES)

# Remove control gates that are forbidden
for gate, gate_name in zip(CONTROL_GATES, CONTROL_GATE_NAMES):
    if gate in FORBIDDEN_CONTROL_GATES:
        CONTROL_GATES.remove(gate)
        CONTROL_GATE_NAMES.remove(gate_name)

# Default gates when no custom gate colors are provided
DEFAULT_GATES_PAINTING = [YGate, HGate, TGate, SGate, SdgGate, IGate]
DEFAULT_GATES_PAINTING_NAMES = ["y", "h", "t", "s", "sdg", "i"]



def create_style(
    gate_colors,  # List of colors or dictionary of colors for each gate
    background,  # Background color
    linecolor,  # Line color
    textcolor="#FFFFFF",  # Text color
    display_qubit_names=False,  # Whether to display qubit names
    display_gate_names=True,  # Whether to display gate names
    custom_definitions=None,  # Custom gate definitions
    gates=GATES,  # List of available gates
    gate_names=GATE_NAMES  # List of gate names
):
    """
    Create a style dictionary for a quantum circuit image.

    Args:
        gate_colors (list or dict): List of colors or dictionary of colors for each gate.
        background (str): Background color.
        linecolor (str): Line color.
        textcolor (str, optional): Text color. Defaults to "#FFFFFF".
        display_qubit_names (bool, optional): Whether to display qubit names. Defaults to False.
        display_gate_names (bool, optional): Whether to display gate names. Defaults to True.
        custom_definitions (dict, optional): Custom gate definitions. Defaults to None.
        gates (list, optional): List of available gates. Defaults to GATES.
        gate_names (list, optional): List of gate names. Defaults to GATE_NAMES.

    Returns:
        tuple: A tuple containing the style dictionary, a list of possible gates, and a list of possible control gates.
    """
    # Create the style dictionary
    custom_style = {
        "displaycolor": {},  # Dictionary of gate colors
        "backgroundcolor": background,  # Background color
        "linecolor": linecolor,  # Line color
        "textcolor": background if not display_qubit_names else textcolor,  # Text color
    }

    # Iterate over gate colors
    if type(gate_colors) == list:
        iterator = zip(gate_colors, gate_names)
    elif type(gate_colors) == dict:
        iterator = gate_colors.items()
    for color, gate_name in iterator:
        # Determine the gate name color
        gate_name_color = textcolor if display_gate_names else color

        # Add gate colors to the style dictionary
        custom_style["displaycolor"][gate_name] = [color, gate_name_color]
        custom_style["displaycolor"][f"c{gate_name}"] = [
            color,
            gate_name_color,
        ]
        custom_style["displaycolor"][f"cc{gate_name}"] = [
            color,
            gate_name_color,
        ]

    # Add custom gate definitions to the style dictionary
    if custom_definitions:
        for k, v in custom_definitions.items():
            custom_style["displaycolor"][k] = v

    # Determine the list of possible gates
    possible_gates = gates[: len(gate_colors)]

    # Determine the list of possible control gates
    possible_control_gates = possible_gates[:]
    for gate in possible_control_gates:
        if gate in FORBIDDEN_CONTROL_GATES:
            possible_control_gates.remove(gate)

    # Return the style dictionary, list of possible gates, and list of possible control gates
    return custom_style, possible_gates, possible_control_gates


def char_to_pixels(
    text, path="ARCADE_N.TTF", size=2, *args, **kwargs
):
    """
    Converts a character to a 2D array of pixels.

    Args:
        text (str): The character to convert.
        path (str, optional): The path to the font file. Defaults to "ARCADE_N.TTF".
        size (int, optional): The size of the font. Defaults to 2 gates/pixel.
        *args: Additional arguments to pass to the `text` method of the `ImageDraw` class.
        **kwargs: Additional keyword arguments to pass to the `text` method of the `ImageDraw` class.

    Returns:
        ndarray: A 2D array of pixels where each pixel is represented by an integer.
    """
    # Convert the text to uppercase
    text = text.upper()

    # Load the font
    font = ImageFont.truetype(path, size*7)

    # Get the bounding box of the text
    l, r, w, h = font.getbbox(text)

    # Double the height of the text
    h *= 2

    # Create a new image with white background
    image = Image.new("L", (w, h), 1)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    draw.text((0, 0), text, font=font, *args, **kwargs)

    # Convert the image to a numpy array
    arr = np.asarray(image)

    # Set all non-zero values to 0
    arr = np.where(arr, 0, 1)

    # Remove the rows with all zeros
    arr = arr[(arr != 0).any(axis=1)]

    # Return the resulting array
    return arr

def add_bleed(image, bleed=3, zero=0):
    """
    Adds a bleed border to an image.

    Args:
        image (ndarray): The input image.
        bleed (int, optional): The size of the bleed border. Defaults to 3.
        zero (int, optional): The value to fill the border with. Defaults to 0.

    Returns:
        ndarray: The image with a bleed border.
    """
    # Get the size of the input image
    imsize = image.shape

    # Calculate the size of the full image with bleed
    full_size = imsize[0] + 2 * bleed, imsize[1] + 2 * bleed

    # Create a new array with the full size and fill it with the zero value
    pixels = np.full(shape=full_size, fill_value=zero)

    # Calculate the starting and ending indices for the input image in the full array
    x_start, x_end = int((pixels.shape[0] / 2) - imsize[0] / 2), int(
        (pixels.shape[0] / 2) + imsize[0] / 2
    )
    y_start, y_end = int((pixels.shape[1] / 2) - imsize[1] / 2), int(
        (pixels.shape[1] / 2) + imsize[1] / 2
    )

    # Copy the input image into the full array
    pixels[x_start:x_end, y_start:y_end] = np.asarray(image)

    # Return the full array
    return pixels


def load_image(path, new_height=None, num_colors=3):
    """
    Load an image from a file path and optionally resize it.

    Args:
        path (str): The path to the image file.
        new_height (int, optional): The desired height of the image. Defaults to None.
        num_colors (int, optional): The number of colors to reduce the image to. Defaults to 3.

    Returns:
        PIL.Image.Image: The loaded and potentially resized image.
    """
    # Load the image in RGB mode and convert it to paletted mode with the specified number of colors
    image = (
        Image.open(path)
        .convert("RGB")
        .convert("P", palette=Image.Palette.ADAPTIVE, colors=num_colors)
    )

    # If a new height is specified, resize the image while maintaining aspect ratio
    if new_height != None:
        width, height = image.size
        new_width = new_height * width / height
        size = (int(new_width), int(new_height))
        image = image.resize(size)

    # Return the loaded and potentially resized image
    return image





def get_colors_from_image(image, zero=0):
    """
    Extracts color information from an image.

    Args:
        image (PIL.Image.Image): The image to extract colors from.
        zero (int, optional): The color to treat as transparent. Defaults to 0.

    Returns:
        tuple: A tuple containing three dictionaries:
            - color_dict: A dictionary mapping image palette colors to hexadecimal color strings.
            - gate_dict: A dictionary mapping image palette colors to gate types.
            - gate_to_color: A dictionary mapping hexadecimal color strings to gate names.
    """
    # Initialize dictionaries to store color information
    color_dict = {}  # Maps image palette colors to hexadecimal color strings
    gate_dict = {}  # Maps image palette colors to gate types
    gate_to_color = {}  # Maps hexadecimal color strings to gate names

    i = 0
    # Iterate over the image's palette colors
    for k, v in image.palette.colors.items():
        # Get the RGB color values of the current palette color
        color_rgb = list(image.palette.colors.keys())[
            list(image.palette.colors.values()).index(v)
        ]
        # Convert the RGB values to a hexadecimal color string
        color_dict[v] = mpl_colors.to_hex([c / 255 for c in k])
        # If the current color is not transparent, add it to the gate dictionaries
        if v != zero:
            gate_dict[v] = DEFAULT_GATES_PAINTING[i]
            gate_to_color[color_dict[v]] = DEFAULT_GATES_PAINTING_NAMES[i]
            i += 1

    # Sort the dictionaries and return them
    color_dict = dict(sorted(color_dict.items()))
    gate_dict = dict(sorted(gate_dict.items()))
    gate_to_color = dict(sorted(gate_to_color.items()))

    return color_dict, gate_dict, gate_to_color



def load_image(path, new_height=None, num_colors=3):
    """
    Load an image from a file path and optionally resize it.

    Args:
        path (str): The path to the image file.
        new_height (int, optional): The desired height of the image. Defaults to None.
        num_colors (int, optional): The number of colors to reduce the image to. Defaults to 3.

    Returns:
        PIL.Image.Image: The loaded and potentially resized image.
    """
    # Load the image in RGB mode and convert it to paletted mode with the specified number of colors
    image = Image.open(path)
    image = image.convert("RGB")
    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=num_colors)

    # If a new height is specified, resize the image while maintaining aspect ratio
    if new_height is not None:
        width, height = image.size
        new_width = new_height * width / height
        size = (int(new_width), int(new_height))
        image = image.resize(size)

    return image


def create_qc_from_pixels(
    pixels,
    seed=None,
    color_mode="random",
    control_probability=0.4,
    gates=GATES,
    control_gates=CONTROL_GATES,
    gate_dict={},
    zero=0,
):
    """
    Create a quantum circuit from a matrix of pixel values.

    Args:
        pixels (numpy.ndarray): The matrix of pixel values.
        seed (int, optional): The random seed. Defaults to None.
        color_mode (str, optional): The color mode. Defaults to "random".
        control_probability (float, optional): The probability of applying a control gate. Defaults to 0.4.
        gates (list, optional): The list of gates. Defaults to GATES.
        control_gates (list, optional): The list of control gates. Defaults to CONTROL_GATES.
        gate_dict (dict, optional): The dictionary of gates. Defaults to {}.
        zero (int, optional): The value representing no pixel. Defaults to 0.

    Returns:
        QuantumCircuit: The quantum circuit.
    """

    def choose_gate(
        px,
        color_mode=color_mode,
        control=False,
        gates=gates,
        control_gates=control_gates,
    ):
        """
        Choose a gate based on the color mode and control flag.

        Args:
            px (int): The pixel value.
            color_mode (str, optional): The color mode. Defaults to color_mode.
            control (bool, optional): Whether to apply a control gate. Defaults to False.
            gates (list, optional): The list of gates. Defaults to gates.
            control_gates (list, optional): The list of control gates. Defaults to control_gates.

        Returns:
            BaseOperator: The chosen gate.
        """
        match (color_mode, control):
            case ("random", False):
                return np.random.choice(gates)()
            case ("random", True):
                return np.random.choice(control_gates)()
            case ("paint", _):
                return gate_dict[px]()
            case _:
                raise ValueError("Invalid color mode or control flag.")

    # Set random seed
    if seed:
        np.random.seed(seed)

    # Create quantum circuit
    num_qubits = pixels.shape[0]
    qc = QuantumCircuit(pixels.shape[0])

    # Iterate over all pixels
    for x, col in enumerate(pixels.T):
        # heights with and without pixels
        true_pixels = np.argwhere(np.array(col != zero, dtype=bool)).flatten()
        false_pixels = np.argwhere(np.array(col == zero, dtype=bool)).flatten()

        # Go over all y's
        col_iter = iter(enumerate(col))
        for y, px in col_iter:
            if px != zero:  # If there's something
                add_control = np.random.random() < control_probability

                if not add_control:
                    # Choose a random gate
                    gate = choose_gate(px)
                    qc.append(gate, [qc.qubits[y]])
                    continue

                else:
                    options = []
                    # Check if there are empty pixels above:
                    if y - 1 in false_pixels:
                        # Randomly choose control qubit
                        if (true_pixels < y).any():
                            start_in_column = np.argwhere(
                                false_pixels < y
                            ).max()
                            options += list(
                                range(
                                    np.argwhere(true_pixels < y).max()
                                    + start_in_column,
                                    y,
                                )
                            )
                        else:
                            options += list(range(0, y))

                    # Check for empty pixels below
                    if y + 1 in false_pixels:
                        # Randomly choose control qubit
                        if (true_pixels > y).any():
                            # Determine range of options
                            next_pixel = np.argwhere(true_pixels == y) + 1
                            next_pixel = true_pixels[next_pixel]
                            options += list(range(y + 1, int(next_pixel)))
                        else:
                            options += list(range(y + 1, num_qubits))

                    if len(options) > 0:
                        # Make sure there's a preference over closer qubits
                        p = np.array(np.abs(np.array(options) - y))
                        p = 1 / p * (1 / sum(1 / p))

                        # Choose control qubit
                        control_qubit = np.random.choice(options, p=p)

                        # Choose gate
                        gate = choose_gate(px, control=True)

                        qc.append(
                            gate.control(),
                            [qc.qubits[control_qubit], qc.qubits[y]],
                        )

                        # Make sure not to cross with controls
                        false_pixels = list(false_pixels)
                        if control_qubit in false_pixels:
                            false_pixels.remove(control_qubit)
                        false_pixels = np.array(false_pixels)

                        true_pixels = list(true_pixels)
                        true_pixels.append(control_qubit)
                        true_pixels = np.array(true_pixels)

                        if y < control_qubit:
                            skip = (
                                np.argmax((false_pixels > y)) - control_qubit
                            )
                            for _ in range(skip):
                                next(col_iter)
                        continue

                    else:
                        # Choose a random gate
                        gate = choose_gate(px)
                        qc.append(gate, [qc.qubits[y]])
                        continue

        if not true_pixels.any():
            options = list(range(num_qubits))
            qubit_1 = np.random.choice(options)

            options.pop(qubit_1)
            # Make sure there's a preference over closer qubits
            p = np.array(np.abs(np.array(options) - qubit_1))
            p = 1 / p * (1 / sum(1 / p))

            qubit_2 = np.random.choice(options, p=p)

            rand = np.random.random()
            if rand < (1/3):
                # Apply CZ
                qc.cz(qubit_1, qubit_2)
            elif rand < (2/3):
                # Apply CNOT
                qc.cx(qubit_1, qubit_2)
            else:
                # Apply SWAP
                qc.swap(qubit_1, qubit_2)

        qc.barrier(qc.qubits)

    return qc
