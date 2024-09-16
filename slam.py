import numpy as np
import matplotlib.pyplot as plt

class SimpleSLAM:
    def __init__(self, grid_size, grid_resolution):
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.map = np.zeros((grid_size, grid_size))
        self.robot_pose = np.array([grid_size // 2, grid_size // 2])  # Start in the center
    
    def move_robot(self, dx, dy):
        self.robot_pose += np.array([dx, dy])
    
    def update_map(self, sensor_data):
        for dist, angle in sensor_data:
            x = int(self.robot_pose[0] + dist * np.cos(angle))
            y = int(self.robot_pose[1] + dist * np.sin(angle))
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.map[x, y] = 1  # Mark as occupied
    
    def plot_map(self):
        plt.imshow(self.map, cmap='gray')
        plt.plot(self.robot_pose[1], self.robot_pose[0], 'bo')  # Robot position
        plt.show()

# Example usage
slam = SimpleSLAM(grid_size=100, grid_resolution=1)
slam.move_robot(5, 5)
sensor_data = [(10, np.pi/4), (8, np.pi/2)]  # Example sensor data (distance, angle)
slam.update_map(sensor_data)
slam.plot_map()
