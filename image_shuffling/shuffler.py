import argparse
import random
import warnings
import pathlib
import cv2 as cv
import numpy as np
from datetime import datetime
import os
import time

CURR_TIME = int(time.time() * 1000)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

class Shuffler:
    def __init__(self, image: str) -> None:
        self.path = pathlib.Path(image)
        self.original = cv.imread(str(self.path.resolve()), cv.IMREAD_UNCHANGED)

        if self.original is None:
            raise ValueError('image is None')

        self.shuffled = self.original.copy()
        self.x = self.original.shape[1]
        self.y = self.original.shape[0]

        self._pieces = []

    def _check_argument(self, matrix: tuple) -> None:
        if len(matrix) != 2 or not all(isinstance(x, int) for x in matrix):
            raise ValueError(
                'matrix must be 2-dimensional containing only integer numbers'
            )
        elif min(matrix) <= 0:
            raise ValueError('matrix values must be greater than 0')
        elif matrix > (self.x, self.y):
            raise ValueError('number of splits greater than pixels')

    def _check_pixel_loss(self, x_missing: int, y_missing: int) -> None:
        new_x = self.x - x_missing
        new_y = self.y - y_missing

        if (x_missing + y_missing) > 0:
            warnings.warn(
                'Splitting images into non-integer intervals causes pixel '
                f'loss. Original Shape: ({self.x}, {self.y}) New Shape: '
                f'({new_x}, {new_y})',
                stacklevel=2,
            )

    def _split(self, x: int, y: int, x_list: list, y_list: list) -> None:
        self._pieces = [
            self.original[y_n * y:y_piece, x_n * x:x_piece]
            for x_n, x_piece in enumerate(x_list)
            for y_n, y_piece in enumerate(y_list)
        ]

    def _generate_image(self, cols: int) -> None:
        self.shuffled = np.hstack(
            [np.vstack(chunk) for chunk in zip(*[iter(self._pieces)] * cols)]
        )

    def shuffle(self, matrix: tuple, method: str = 'random', grid: list = None, tile_mapping: dict = None) -> None:
        self._check_argument(matrix)

        x = self.x // matrix[1]
        x_missing = self.x % matrix[1]
        x_list = list(range(x, self.x + 1, x))

        y = self.y // matrix[0]
        y_missing = self.y % matrix[0]
        y_list = list(range(y, self.y + 1, y))

        self._check_pixel_loss(x_missing, y_missing)
        self._split(x, y, x_list, y_list)

        if tile_mapping:
            new_pieces = self._pieces.copy()  # Copy the original pieces
            for input_tile, output_tile in tile_mapping.items():
                input_index = input_tile[1] * matrix[1] + input_tile[0]
                output_index = output_tile[1] * matrix[1] + output_tile[0]
                if input_index < len(self._pieces) and output_index < len(new_pieces):
                    new_pieces[output_index] = self._pieces[input_index]
                else:
                    warnings.warn(f"Invalid tile mapping: {input_tile} -> {output_tile}. Ignoring this mapping.")

            self._pieces = new_pieces
        else:
            folder_path = os.path.dirname(self.path)
            seed = hash(folder_path) + CURR_TIME
            random.seed(seed)

            if grid:
                try:
                    self._pieces = [self._pieces[i] for i in grid]
                except IndexError:
                    warnings.warn('Invalid grid provided. Shuffling randomly.')
                    random.shuffle(self._pieces)
            else:
                if method == 'random':
                    random.shuffle(self._pieces)
                elif method == 'reverse':
                    self._pieces = self._pieces[::-1]
                elif method == 'rotate':
                    self._pieces = self._pieces[1:] + self._pieces[:1]
                else:
                    raise ValueError(f'Unknown method: {method}')

        self._generate_image(matrix[0])

    def show(self) -> None:
        cv.imshow('Image', self.shuffled)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save(self, matrix: tuple, grid: list = None, output_dir: pathlib.Path = None, grid_lines: bool = False, grid_lines_thickness: int = 10, grid_rows: int = None, grid_cols: int = None, include_timestamp: bool = False, include_row_grid_info: bool = False) -> None:
        grid_str = f"_grid{''.join(map(str, grid))}" if grid else ""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if include_timestamp else ""
        row_grid_info = f"_rows{matrix[0]}_cols{matrix[1]}" if include_row_grid_info else ""
        
        new_filename = f"{self.path.stem}{row_grid_info}{grid_str}{timestamp}{self.path.suffix}"
        output_path = output_dir / new_filename if output_dir else self.path.parent.resolve() / new_filename

        if grid_lines:
            if grid_rows and grid_cols:
                self.shuffled = self.draw_lines_at_intersections((grid_rows, grid_cols), grid_lines_thickness)
            else:
                self.shuffled = self.draw_lines_at_intersections(matrix, grid_lines_thickness)

        cv.imwrite(str(output_path), self.shuffled)

        if "RGB" in self.path.stem:
            # Extract the numeric identifier from the filename
            parts = self.path.stem.split('_')

            grasps_filename = ""
            if "pen" in self.path.stem:
                numeric_identifier = parts[2]
                grasps_filename = f"{parts[0]}_{parts[1]}_{numeric_identifier}_grasps.txt"

            else:
                numeric_identifier = parts[1]
                grasps_filename = f"{parts[0]}_{numeric_identifier}_grasps.txt"

            grasps_output_path = output_dir / grasps_filename if output_dir else self.path.parent.resolve() / grasps_filename

            # Determine the path of the original "grasps.txt" file
            original_grasps_path = self.path.parent / grasps_filename

            # Copy contents of the original "grasps.txt" file to the new location
            with open(original_grasps_path, 'r') as original_file:
                contents = original_file.read()

            with open(grasps_output_path, 'w') as new_file:
                new_file.write(contents)

    def draw_lines_at_intersections(self, matrix: tuple, grid_lines_thickness: int):
        height, width, *_ = self.shuffled.shape
        
        # Calculate tile sizes based on matrix
        tile_width = width // matrix[1]
        tile_height = height // matrix[0]
        
        # Draw vertical lines
        for x in range(0, width, tile_width):
            cv.line(self.shuffled, (x, 0), (x, height), (0, 0, 0), grid_lines_thickness)
        
        # Draw horizontal lines
        for y in range(0, height, tile_height):
            cv.line(self.shuffled, (0, y), (width, y), (0, 0, 0), grid_lines_thickness)
        
        return self.shuffled

def main():
    parser = argparse.ArgumentParser(description='Shuffle an image.')
    parser.add_argument('--input_dir', '-i', type=str, help='Path to the input directory.')
    parser.add_argument('--output_dir', '-o', type=str, default='.', help='Path to the output directory.')
    parser.add_argument('--image_path', '-p', type=str, help='Path to a single image.')
    parser.add_argument('--rows', '-r', type=int, default=1, help='Number of rows to split the image into.')
    parser.add_argument('--cols', '-c', type=int, default=1, help='Number of columns to split the image into.')
    parser.add_argument('--method', '-m', type=str, default='random', help='Method to rearrange the tiles.')
    parser.add_argument('--grid', '-g', type=int, nargs='*', help='Custom grid of tiles.')
    parser.add_argument('--config', '-con', type=str, help='Path to the config file.')
    parser.add_argument('--tile_mapping', '-t', type=str, help='Path to the tile mapping file.')
    parser.add_argument('--show', '-sh', action='store_true', help='Show the shuffled image.')
    parser.add_argument('--save', '-s', action='store_true', help='Save the shuffled image.')
    parser.add_argument('--include-timestamp', '-it', action='store_true', help='Include timestamp in the filename')
    parser.add_argument('--include-row-grid-info', '-irg', action='store_true', help='Include row and grid info in the filename')

    parser.add_argument('--grid_lines', '-l', action='store_true', help='Show grid lines.')
    parser.add_argument('--grid_lines_thickness', '-lt', type=int, default=10, help='Thickness of grid lines.')
    parser.add_argument('--grid_rows', '-gr', type=int, default=None, help='Custom number of rows for grid lines.')
    parser.add_argument('--grid_cols', '-gc', type=int, default=None, help='Custom number of cols for grid lines.')
    args = parser.parse_args()

    grid = args.grid
    if args.config:
        with open(args.config, 'r') as f:
            grid = [int(line.strip()) for line in f]

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_mapping = None
    if args.tile_mapping:
        with open(args.tile_mapping, 'r') as f:
            tile_mapping = {}
            for line in f:
                input_tile_str, output_tile_str = line.strip().split(':')
                input_tile = tuple(map(int, input_tile_str.split(',')))
                output_tile = tuple(map(int, output_tile_str.split(',')))
                tile_mapping[input_tile] = output_tile

    grid_lines = args.grid_lines
    grid_lines_thickness = args.grid_lines_thickness
    grid_rows = args.grid_rows
    grid_cols = args.grid_cols

    if args.image_path:
        image_path = pathlib.Path(args.image_path)
        image = Shuffler(str(image_path))
        image.shuffle(matrix=(args.rows, args.cols), method=args.method, grid=grid, tile_mapping=tile_mapping)

        if args.show:
            image.show()

        if args.save:
            image.save(matrix=(args.rows, args.cols), grid=grid, output_dir=output_dir, grid_lines=grid_lines, grid_lines_thickness=grid_lines_thickness, grid_rows=grid_rows, grid_cols=grid_cols, include_timestamp=args.include_timestamp, include_row_grid_info=args.include_row_grid_info)
    else:
        input_dir = pathlib.Path(args.input_dir)

        for image_path in input_dir.rglob('*'):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image = Shuffler(str(image_path))
                image.shuffle(matrix=(args.rows, args.cols), method=args.method, grid=grid, tile_mapping=tile_mapping)

                if args.show:
                    image.show()

                if args.save:
                    # Construct the relative path and the corresponding output path
                    relative_path = image_path.relative_to(input_dir)
                    output_path = output_dir / relative_path.parent
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    image.save(matrix=(args.rows, args.cols), grid=grid, output_dir=output_path, grid_lines=grid_lines, grid_lines_thickness=grid_lines_thickness, grid_rows=grid_rows, grid_cols=grid_cols, include_timestamp=args.include_timestamp, include_row_grid_info=args.include_row_grid_info)


if __name__ == "__main__":
    main()