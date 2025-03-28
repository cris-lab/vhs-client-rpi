import cv2

class Label:
    
    def __init__(self, text, position, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=2, color=(0, 255, 0), thickness=2, padding=10, background_color=None):
        self.text = text
        self.position = position
        self.font = font
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness
        self.padding = padding
        self.background_color = background_color


    def draw(self, frame):
    # Calcular el tamaño del texto y el rectángulo
        (text_width, text_height), baseline = cv2.getTextSize(self.text, self.font, self.font_scale, self.thickness)

        # Ajustar la posición del rectángulo para que quede completamente dentro del fondo
        rectangle_start = (self.position[0] - self.padding, self.position[1] - text_height - baseline + self.padding)
        rectangle_end = (self.position[0] + text_width + self.padding, self.position[1] + baseline + self.padding)

        if self.background_color is None:
            # Dibujar el rectángulo blanco como fondo
            cv2.rectangle(frame, rectangle_start, rectangle_end, (255, 255, 255), cv2.FILLED)

        # Ajustar la posición del texto para centrarlo dentro del rectángulo
        text_position_y = self.position[1] + baseline  # Ajustar la posición Y para que el texto quede centrado

        # Escribir el texto en el frame
        cv2.putText(frame, self.text, (self.position[0], text_position_y), self.font, self.font_scale, self.color, self.thickness)


        
        
