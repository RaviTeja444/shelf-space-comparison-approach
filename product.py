
class Product:
    def __init__(self, id, name, category, width, profit, max_facings):
        self.id = id
        self.name = name
        self.category = category
        self.width = width  # in cm
        self.profit = profit  # profit per facing
        self.max_facings = max_facings
    
    def __repr__(self):
        return f"Product({self.id}: {self.name}, w={self.width}cm, p=${self.profit})"


class Result:
    def __init__(self, method_name, allocations, total_profit, total_width, computation_time):
        self.method = method_name
        self.allocations = allocations  # dict: product_id -> number of facings
        self.total_profit = total_profit
        self.total_width = total_width
        self.computation_time = computation_time
    
    def get_utilization(self, capacity):
        return (self.total_width / capacity) * 100