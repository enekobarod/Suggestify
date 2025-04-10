import tkinter as tk
from PIL import Image, ImageTk
from model_ratings import RecommenderModel
import os
import pandas as pd

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interfaz de Gestos")
        
        self.user_id= 1
        try:
            with open("register.csv", "r") as f:
                lineas = f.readlines()
                if lineas:
                    ultima_linea = lineas[-1].strip()
                    columnas = ultima_linea.split(",")
                    if columnas and columnas[0].strip().isdigit():
                        self.user_id = int(columnas[0].strip())+1

        except FileNotFoundError:
            with open("register.csv", "w") as f:
                f.write(f"")
                pass
        except Exception as e:
            print(f"Error al leer el archivo: {e}")

        print("user_id =", self.user_id)

        #tamaño fijo
        self.root.geometry("400x700")
        self.root.resizable(False, False)
        
        self.model = RecommenderModel(min_ratings_needed=20)
        self.current_track = None
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.animation_in_progress = False
        self.title_animation_id = None
        self.title_position = 0
        
        self.scroll_step = 0.5  #step size on title scroll
        self.scroll_delay = 400  #step rhythm
        
        #paleta de colores
        self.COLOR_PRINCIPAL = "#2E3440"
        self.COLOR_SECUNDARIO = "#434C5E"
        self.COLOR_TEXTO = "#ECEFF4"
        self.COLOR_ACENTO = "#88C0D0"
        self.COLOR_FEEDBACK = "#BF616A"

        root.configure(bg=self.COLOR_PRINCIPAL)
        
        #hueco pa la imagen
        self.canvas = tk.Canvas(
            root, 
            width=400, 
            height=600, 
            bg=self.COLOR_PRINCIPAL,
            highlightthickness=0
        )
        self.canvas.pack()

        text_frame = tk.Frame(self.root, height=100, width=400)
        text_frame.pack_propagate(False)
        text_frame.pack()
        
        #etiquetas con desplazamiento automático
        self.title_label = tk.Label(text_frame, text="", font=("Arial", 20), wraplength=380, justify="center")
        self.title_label.pack(pady=(10, 0))
        self.author_label = tk.Label(text_frame, text="", font=("Arial", 16))
        self.author_label.pack()

        #events
        self.canvas.bind("<Button-1>", self.on_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        self.show_new_track()
        self.start_title_scroll()


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        plot_folder = os.path.join(os.getcwd(), "plots")
        os.makedirs(plot_folder, exist_ok=True)

        user_pca_file = os.path.join(plot_folder, f"user_{self.user_id}_pca.png")
        self.model.save_user_plot(user_pca_file)

        last_user_id = self.user_id - 1
        user_likes_file = os.path.join(plot_folder, f"user_{last_user_id}_likes.png")
        self.model.save_user_like_evolution_plot(last_user_id, user_likes_file)

        self.root.destroy()       

    def show_new_track(self):
        #mostrar nueva canción
        if self.model.has_enough_ratings():
            self.show_recommendations()
            return

        track_info = self.model.get_popular_song()
        self.current_track = track_info
        self.title_text = track_info["track_name"]+ "  " #espacios para separar final y principio

        #ponerle caratula
        #self.image = Image.open("caratula.jpg").resize((400, 600), Image.Resampling.LANCZOS)
        self.image = self.current_track["cover_image"].resize((400, 600), Image.Resampling.LANCZOS)
        self.img_display = ImageTk.PhotoImage(self.image)

        self.canvas.delete("all")
        self.track_image = self.canvas.create_image(200, 300, image=self.img_display)

        #update de titulo y autor
        self.title_label.config(text=track_info["track_name"])
        self.author_label.config(text=track_info["artist_name"])
        self.start_title_scroll()

    def start_title_scroll(self):
        #teleprompter en el titulo
        if self.title_animation_id:
            self.root.after_cancel(self.title_animation_id)
        
        self.title_position = 0.0
        self.animate_title_scroll()

    def animate_title_scroll(self):
        #animacion teleprompter
        display_length = 20
        if len(self.title_text) > display_length:
            self.title_position += self.scroll_step 
            
            #volver a empezar
            if self.title_position > len(self.title_text):
                self.title_position = 0

            start_pos = int(self.title_position)
            partial = self.title_position - start_pos
            
            visible_text = self.title_text[start_pos:] + self.title_text[:start_pos]
            if len(visible_text) > display_length + 1:
                next_char = visible_text[display_length] if len(visible_text) > display_length else visible_text[0]
                blended_char = self.blend_chars(visible_text[display_length-1], next_char, partial)
                visible_text = visible_text[:display_length-1] + blended_char

            self.title_label.config(text=visible_text[:display_length])

        self.root.after(self.scroll_delay, self.animate_title_scroll)

    def blend_chars(self, char1, char2, ratio):
        if ratio < 0.3:
            return char1
        elif ratio > 0.7:
            return char2
        else:
            return char1 

    def on_start(self, event):
        if self.animation_in_progress:
            return
            
        self.drag_data = {"x": event.x, "y": event.y, "item": self.track_image}

    def on_drag(self, event):
        if self.animation_in_progress or not self.drag_data["item"]:
            return
            
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        
        self.canvas.move(self.drag_data["item"], dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def show_feedback_animation(self, direction):
        #animacion de gesto
        self.canvas.delete("feedback")
        
        img_map = {
            'right': 'heart.png',
            'left': 'x_mark.png',
            'up': 'jump_icon.png',
            'double_click': 'heart.png'
        }
        
        try:
            img = Image.open(img_map[direction])
            TARGET_SIZE = (100, 100)
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            feedback_img = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return
        
        self.feedback_img = feedback_img 
        self.canvas.create_image(
            200, 300,
            image=self.feedback_img,
            tags="feedback",
            anchor="center"
        )
        self.root.after(500, lambda: self.canvas.delete("feedback"))

    def show_double_hearts_feedback(self):
        self.canvas.delete("feedback")
        try:
            heart_img = Image.open("heart.png")
            TARGET_SIZE = (80, 80)
            heart_img = heart_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            #rotar. no me gusta, queda bien cutre pero bueno :/
            heart_left = heart_img.rotate(15)
            heart_right = heart_img.rotate(-15)
            
            feedback_img_left = ImageTk.PhotoImage(heart_left)
            feedback_img_right = ImageTk.PhotoImage(heart_right)
            
            self.feedback_img_left = feedback_img_left
            self.feedback_img_right = feedback_img_right
            
            self.canvas.create_image(
                200 - 50, 300,
                image=feedback_img_left,
                tags="feedback",
                anchor="center"
            )
            self.canvas.create_image(
                200 + 50, 300,
                image=feedback_img_right,
                tags="feedback",
                anchor="center"
            )
            self.root.after(500, lambda: self.canvas.delete("feedback"))
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return

    def on_release(self, event):
        #fin de gesto
        if self.animation_in_progress:
            return
            
        pos = self.canvas.coords(self.track_image)
        if not pos:
            return
            
        x, y = pos
        gesture = None
        
        #determinar gesto
        if x > 400:
            gesture = 'right'
        elif x < 0:
            gesture = 'left'
        elif y < 150:
            gesture = 'up'
            
        if gesture:
            self.show_feedback_animation(gesture)
            self.root.after(500, lambda: self.handle_gesture(gesture))
        else:
            self.animate_return_to_center()

    def on_double_click(self, event):
        if self.animation_in_progress:
            return
        self.show_double_hearts_feedback()
        self.root.after(500, lambda: self.handle_gesture('double_click'))

    def animate_return_to_center(self):
        #suavidad en movimientos
        self.animation_in_progress = True
        
        def update_animation():
            pos = self.canvas.coords(self.track_image)
            dx = 200 - pos[0]
            dy = 300 - pos[1]
            
            if abs(dx) > 1 or abs(dy) > 1:
                self.canvas.move(self.track_image, dx*0.2, dy*0.2)
                self.root.after(16, update_animation)
            else:
                self.canvas.coords(self.track_image, 200, 300)
                self.animation_in_progress = False
        
        update_animation()

    def handle_gesture(self, gesture):
        #gesto
        if self.current_track is None:
            return
            
        self.model.submit_gesture_rating(self.user_id, self.current_track["track_id"], gesture)
        
        if self.model.has_enough_ratings():
            self.show_recommendations()
        else:
            self.show_new_track()

    def show_recommendations(self):
        recs = self.model.get_final_recommendations(top_n=5, diversity=0.3)
        
        if len(recs) > 0:
            #recomoendacion
            first_rec = recs.iloc[0]
            self.current_track = {
                "track_id": first_rec["track_id"],
                "track_name": first_rec["track_name"],
                "artist_name": first_rec["artist_name"]
            }
            self.title_text = first_rec["track_name"] + "  "  #espacios para separación de final y principio

            #notificar recomendacion
            print(f"\nRECOMMENDED TRACK: {first_rec['track_name']} by {first_rec['artist_name']}")

            #self.image = Image.open("caratula.jpg").resize((400, 600), Image.Resampling.LANCZOS)
            image = self.model.get_cover_image(first_rec["track_name"], first_rec["artist_name"])
            self.image = image.resize((400, 600), Image.Resampling.LANCZOS)
            self.img_display = ImageTk.PhotoImage(self.image)

            #actualizar
            self.canvas.delete("all")
            self.track_image = self.canvas.create_image(200, 300, image=self.img_display)
            self.title_label.config(text=first_rec["track_name"])
            self.author_label.config(text=first_rec["artist_name"])
            self.start_title_scroll()
        else:
            self.title_label.config(text="No hay recomendaciones")
            self.author_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()


