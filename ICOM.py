from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import fast_hdbscan
import py4DSTEM
from py4DSTEM.process.diffraction.digital_dark_field import pointlist_to_array
from joblib import Parallel, delayed


class Icom:
    c: list = None            # Colour-labels
    d_mat: np.ndarray = None  # 2D matrix of differences between scan positions

    def __init__(self, directory: str, filename: str, bps: np.ndarray = None) -> Icom:
        '''
        Parameters
        ----------

        directory:
            Filepath to save/load data
        filename:
            Name of .h5 file with bragg-peak data
        bps:
            Optional parameter, an existing array of bragg peak data with rows
            in the form: [Rx, Ry, Intensity, Qx, Qy]
        '''
        self.directory = directory
        self.bparray = bps if bps is not None else self._analyse_file(filename)
        self._get_shape()
        self.set_scan_range(0)

    # ------------------|| Helper and preprocessing methods ||-----------------

    def _analyse_file(self, filename):
        '''Reads data from a .hdf5 and outputs an array of bragg points'''

        py4DSTEM.print_h5_tree(self.directory+filename)

        bragg_peaks = py4DSTEM.read(
            self.directory + filename,
            root='/datacube_root/braggvectors'
        )

        bparray = pointlist_to_array(
            bragg_peaks,
            center=True,
            ellipse=None,
            pixel=True,
            rotate=True,
            rphi=False,
        )

        return bparray

    def _get_shape(self) -> (int, int):
        '''
        Returns the no. of (rows, cols) for self.bparray,
        the full set of scan points
        '''
        self.rows = np.max(self.bparray[:, 3]).astype('int')+1
        self.cols = np.max(self.bparray[:, 4]).astype('int')+1
        return self.rows, self.cols

    def _get_scan_shape(self) -> (int, int):
        '''
        Returns the no. of (rows, cols) self.scan_bps,
        scan points within the limits defined in set_scan_range()
        '''
        return self.scan_rows, self.scan_cols

    def set_scan_range(self, min_x, max_x=-1, min_y=0, max_y=-1):
        '''
        Sets the range of scan points to search
        '''
        self.min_x = min_x
        self.max_x = max_x if max_x != -1 else self.rows
        self.min_y = min_y
        self.max_y = max_y if max_y != -1 else self.cols

        self.scan_rows = self.max_x-self.min_x
        self.scan_cols = self.max_y-self.min_y

        scan_bps = self.bparray[
            (self.bparray[:, 3] >= self.min_x) &
            (self.bparray[:, 3] < self.max_x) &
            (self.bparray[:, 4] >= self.min_y) &
            (self.bparray[:, 4] < self.max_y)
        ]

        scan_bps[:, 3] -= min_x
        scan_bps[:, 4] -= min_y

        self.scan_bps = scan_bps

    # def points_by_cluster(self):
    #     np.unique(self.c)

    # ------------------|| Methods for Brute Force approach ||-----------------

    # TODO: Consider np.nonzero
    def _point_n_peaks(self, Rx, Ry, n: int) -> np.ndarray:
        '''
        Finds the brightest n peaks for a given Rx, Ry scan point and
        returns an array of n [Qx, Qy] vectors
        Fills with [0,0] entries if too few spots present
        '''
        indices = np.where(np.logical_and(
            self.scan_bps[:, 3] == Rx,
            self.scan_bps[:, 4] == Ry
        ))[0][1:n+1]

        peaks = self.scan_bps[indices][:, :2]
        m = peaks.shape[0]

        while m < n:
            peaks = np.vstack((peaks, np.array([0, 0])))
            m += 1

        return peaks

    # TODO: See if for loops necessary
    def _peaks_to_compare(self, n: int, R_shape: tuple) -> np.ndarray:
        '''
        Returns an array of the n brightest diffraction spots for each
        Rx, Ry position as a flattened 1D array.

        The array is filled with [0,0] entries if too few spots are detected
        at a scan point

        Parameters
        ----------

        n:
            Number of peaks to compare
        '''
        matrix = np.empty(shape=(*R_shape, n, 2))

        for Rx in np.arange(R_shape[0]):
            for Ry in np.arange(R_shape[1]):
                matrix[Rx, Ry] = self._point_n_peaks(Rx, Ry, n)

        matrix_flat = matrix.reshape((
            matrix.shape[0]*matrix.shape[1],
            matrix.shape[2],
            matrix.shape[3]
        ))

        return matrix_flat

    def _Rfilter_pointarray(self, Rx: int, Ry: int) -> np.ndarray:
        '''
        Determines indices of rows in the pointarray where the
        Rx and Ry values are the chosen ones and returns the
        pointarray sliced with these indices
        '''
        indices = np.where(
            np.logical_and(self.scan_bps[:, 3] == Rx, self.scan_bps[:, 4] == Ry)
        )

        return self.scan_bps[indices]

    def _dist_from_scan_point(self, comparisonarray, Rx: int, Ry: int) -> np.ndarray:
        '''
        Works out pattern distances from one diffraction pattern for
        point Rx, Ry in pointarray to every Rx, Ry point in
        comparisonarray
        '''

        scan_point_peaks = self._Rfilter_pointarray(Rx, Ry)[:, :2]
        subtraction = comparisonarray[:, np.newaxis, :, :] - scan_point_peaks[:, np.newaxis, :]
        distances = np.sqrt((subtraction**2).sum(axis=3))
        distances_min = distances.min(axis=1)
        pattern_distance = distances_min.sum(axis=1)
        return pattern_distance

    def calc_dist_matrix_brute(self, n: int = 5, n_jobs: int = -1) -> np.ndarray:
        '''
        Computes the similarity of scan-positions by the sum of the
        distances between the n-closest peaks. Each job compares a subset of
        scan-positions to every scan position.

        Parameters
        ----------

        n:
            Number of peaks to compare. Significantly increases runtime
            as n increases. Recommended value ~5.
        n_jobs:
            Number of threads/jobs to split the data into. This introduces
            parallelism, running computations simultaneaously and
            significantly reducing run time.

            Leave blank/set to -1 to utilise all logical processors*2.
        '''
        if n_jobs == -1:
            n_jobs = 2*os.cpu_count()

        R_shape = self._get_scan_shape()
        to_compare = self._peaks_to_compare(n, R_shape)
        size = to_compare.shape[0]
        d_mat = np.empty(shape=(size, size))

        def calc_chunk(indices):
            d_chunk = []
            for Rx in indices:
                for Ry in range(R_shape[1]):
                    arrayline = self._dist_from_scan_point(to_compare, Rx, Ry)
                    ind = Ry + Rx*R_shape[1]
                    d_chunk.append((ind, arrayline))
            return d_chunk

        # Construct lists of indices each job should work on
        indices = np.array_split(np.arange(R_shape[0]), n_jobs)

        # Runs each job with each job's parameters and returns
        # a list from each job
        results = Parallel(n_jobs)(
            delayed(calc_chunk)(indices[i]) for i in range(n_jobs)
        )

        # Combine output from each job
        for result in results:
            for output in result:
                d_mat[output[0], :] = output[1]

        self.d_mat = d_mat
        return d_mat

    # ____________________¦¦ End of Brute Force approach ¦¦____________________
    def cluster_brute_force(self, hdb_cluster=None, n_jobs=-1):
        '''
        '''
        if not hdb_cluster:
            hdb_cluster = fast_hdbscan.HDBSCAN(
                min_cluster_size=100,
                n_jobs=n_jobs,
                cluster_selection_epsilon=.1,
            )
        hdb_cluster.fit(self.d_mat)
        self.c = hdb_cluster.labels_

    def generate_random_colormap(self, grid):
        seed = sum(ord(char) for char in 'SusAmogus')
        unique_values = np.unique(grid)
        num_unique = len(unique_values)

        np.random.seed(seed)
        random_colors = np.random.rand(num_unique, 3)

        color_image = np.zeros((grid.shape[0], grid.shape[1], 3))

        for idx, value in enumerate(unique_values):
            color_image[grid == value] = random_colors[idx]

        return color_image

    def plot_brute_cluster(self):
        rows = self.scan_rows
        cols = self.scan_cols

        image_array = (self.c).reshape((rows, cols))

        plt.imshow(self.generate_random_colormap(image_array))
        plt.title('Clustered Crystal')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

    # --------------|| Methods for Principle Component approach ||-------------
    def _point_n_peaks_pca(self, Rx, Ry, n: int) -> np.ndarray:
        '''
        Finds the brightest n peaks for a given Rx, Ry scan point and
        returns an array of n [Qx, Qy] vectors
        Fills with [0,0] entries if too few spots present
        '''
        indices = np.where(np.logical_and(
            self.scan_bps[:, 3] == Rx,
            self.scan_bps[:, 4] == Ry
        ))[0][1:n+1]

        peaks = self.scan_bps[indices][:, :2]
        m = peaks.shape[0]

        while m < n:
            peaks = np.vstack((peaks, np.array([0, 0])))
            m += 1

        return peaks

    # TODO: See if for loops necessary
    def _peaks_to_compare_pca(self, n: int, R_shape: tuple, Rs_to_compare: list) -> np.ndarray:
        '''
        Returns an array of the n brightest diffraction spots for each
        Rx, Ry position as a flattened 1D array.

        The array is filled with [0,0] entries if too few spots are detected
        at a scan point

        Parameters
        ----------

        n:
            Number of peaks to compare
        '''
        matrix = np.zeros(shape=(len(Rs_to_compare), n, 2))

        for i, R in enumerate(Rs_to_compare):
            matrix[i] = self._point_n_peaks_pca(R[0], R[1], n)

        return matrix

    def calc_dist_pca(self, n=5, n_jobs=-1, cams=[[0, 0]],):
        '''
        Computes the similarity of scan-positions by the sum of the
        distances between the n-closest peaks. Each job compares a subset of
        scan-positions to every scan position.

        Parameters
        ----------

        n:
            Number of peaks to compare. Significantly increases runtime
            as n increases. Recommended value ~5.
        n_jobs:
            Number of threads/jobs to split the data into. This introduces
            parallelism, running computations simultaneaously and
            significantly reducing run time.

            Leave blank/set to -1 to utilise all logical processors*2.
        cams:
            List of [Rx, Ry] points to compare to every point.
        '''
        if n_jobs == -1:
            n_jobs = os.cpu_count()

        rows, cols = self.scan_rows, self.scan_cols
        R_shape = (rows, cols)

        to_compare = self._peaks_to_compare_pca(n, R_shape, cams)
        d_mat = np.zeros(shape=(rows*cols, len(cams)+2))

        def calc_chunk(indices):
            d_chunk = []
            for Rx in indices:
                n_rx = Rx/cols
                for Ry in range(cols):
                    n_ry = Ry/cols
                    arrayline = self._dist_from_scan_point(to_compare, Rx, Ry)
                    arrayline = np.append(arrayline, [n_rx, n_ry])
                    ind = Ry + Rx * cols
                    d_chunk.append((ind, arrayline))

            return d_chunk

        indices = np.array_split(np.arange(rows), n_jobs)

        results = Parallel(n_jobs)(
            delayed(calc_chunk)(indices[i]) for i in range(n_jobs)
        )

        for result in results:
            for output in result:
                d_mat[output[0], :] = output[1]

        self.d_mat = d_mat

    def cluster_pca(self):
        hdb_cluster = fast_hdbscan.HDBSCAN(
            min_cluster_size=10,
            n_jobs=-1,
            cluster_selection_epsilon=0.08
        )

        hdb_cluster.fit(self.d_mat)
        self.c = hdb_cluster.labels_

    # ________________¦¦ End of Principle Component approach ¦¦________________

    def plot_pca(self, figscale=(20, 20)):
        matrix = self.d_mat
        rows = self.scan_rows
        cols = self.scan_cols
        plt.figure(figsize=(cols/figscale[0], rows/figscale[1]))
        plt.scatter(matrix[:, -1]*cols, matrix[::-1, -2]*cols, c=self.c, marker='s', s=4)
        plt.show()

    def plot_pca_random_c(self):
        rows = np.max(self.scan_bps[:, 3]).astype('int')+1
        cols = np.max(self.scan_bps[:, 4]).astype('int')+1

        image_array = (self.c).reshape((rows, cols))

        plt.imshow(self.generate_random_colormap(image_array))
        plt.title('Clustered Crystal')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

    def uniq_labels(self):
        unique_labels, all_cluster_sizes = np.unique(self.c, return_counts=True)
        print("Labels:", unique_labels)

        # all_labels = tatio.c.reshape(ori.shape)
        n_clusters = unique_labels.size - 1
        print("Number of clusters:", n_clusters)
        return unique_labels, all_cluster_sizes

    def peaks_by_cluster(self):
        unique_labels, all_cluster_sizes = self.uniq_labels()
        cs = unique_labels[np.argsort(all_cluster_sizes)][::-1]

        output = []
        for c in cs:
            output.append(self.d_mat[self.c == c])

        return output

    def plot_by_cluster(self, figscale=(20, 20)):
        matrix = self.d_mat
        rows = self.scan_rows
        cols = self.scan_cols
        unique_labels, all_cluster_sizes = self.uniq_labels()

        for c in unique_labels[np.argsort(all_cluster_sizes)][::-1]:
            plt.figure(figsize=(cols/figscale[0], rows/figscale[1]))
            plt.xlim(0, cols)
            plt.ylim(0, rows)
            filtered_matrix = matrix[self.c == c]
            plt.scatter(filtered_matrix[:, -1]*cols, filtered_matrix[::-1, -2]*cols, marker='s', s=5)
            plt.title(f"cluster {c}")
            plt.show()

    # -----------------------|| Saving/Loading methods ||----------------------
    def save_d(self, file_name: str = 'd_mat.py'):
        assert self.d_mat is not None, 'd_mat not computed'
        np.save(file_name, self.d_mat)

    def save_c(self, file_name: str = 'c_labels.py'):
        assert self.c is not None, 'c not computed'
        np.save(file_name, self.c)

    def load_d(self, file_name='d_mat.py') -> np.ndarray:
        self.d_mat = np.load(file_name)
        return self.d_mat

    def load_c(self, file_name='c_labels.py') -> np.ndarray:
        self.c = np.load(file_name)
        return self.c