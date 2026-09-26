import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import cv2
import numpy as np
from plate_cleaner import PlateCleaner

# Global cleaner for worker processes to avoid reloading models on every image
_worker_cleaner = None

def get_cleaner():
    global _worker_cleaner
    if _worker_cleaner is None:
        _worker_cleaner = PlateCleaner()
    return _worker_cleaner

def process_single_file(args_tuple):
    input_path, output_path, save_comparison = args_tuple
    try:
        cleaner = get_cleaner()
        orig_bgr, mask_vis, res_bgr, boxes = cleaner.process_image(input_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if save_comparison:
            # Generate 3-panel comparison
            h, w = orig_bgr.shape[:2]
            overlay = orig_bgr.copy()
            mask_pixels = np.any(mask_vis > 0, axis=-1)
            overlay[mask_pixels] = cv2.addWeighted(orig_bgr, 0.4, mask_vis, 0.6, 0)[mask_pixels]
            for b in boxes:
                cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

            max_h = 720
            if h > max_h:
                scale = max_h / h
                nw, nh = int(w * scale), int(h * scale)
                p1 = cv2.resize(orig_bgr, (nw, nh))
                p2 = cv2.resize(overlay, (nw, nh))
                p3 = cv2.resize(res_bgr, (nw, nh))
            else:
                p1, p2, p3 = orig_bgr, overlay, res_bgr

            for img, text in [(p1, "ORIGINAL"), (p2, "DETECCAO E MASCARA"), (p3, "RESULTADO FINAL")]:
                cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            comp_rgb = cv2.cvtColor(np.hstack([p1, p2, p3]), cv2.COLOR_BGR2RGB)
            Image.fromarray(comp_rgb).save(output_path, quality=92)
        else:
            # Save clean final image only
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(res_rgb).save(output_path, quality=95)

        return True, os.path.basename(input_path), len(boxes)
    except Exception as e:
        return False, os.path.basename(input_path), str(e)

def main():
    parser = argparse.ArgumentParser(description="Removedor Automatizado de Caracteres de Placas em Modelos 3D")
    parser.add_argument("--input", "-i", help="Caminho para uma única imagem a ser processada")
    parser.add_argument("--input-dir", "-d", help="Caminho para a pasta com as imagens a serem processadas")
    parser.add_argument("--output-dir", "-o", default="Imagens_Limpa_Placas", help="Pasta de saída para salvar as imagens processadas")
    parser.add_argument("--preview", action="store_true", help="Gera imagem comparativa lado a lado em vez de apenas a imagem final")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Número de processos paralelos (padrão: 4)")

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        print("Por favor, especifique --input [arquivo] ou --input-dir [pasta].")
        print("Exemplo: python run_plate_cleaner.py --input-dir 'Imagens Referência com Placa' --output-dir 'Resultado'")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        filename = os.path.basename(args.input)
        out_path = os.path.join(args.output_dir, f"limpo_{filename}" if not args.preview else f"comparacao_{filename}")
        print(f"Processando imagem individual: {filename}...")
        t0 = time.time()
        ok, name, count_or_err = process_single_file((args.input, out_path, args.preview))
        dt = time.time() - t0
        if ok:
            print(f"[CONCLUÍDO] {name}: {count_or_err} placa(s) encontrada(s) e limpa(s) em {dt:.2f}s.")
            print(f"Arquivo salvo em: {out_path}")
        else:
            print(f"[ERRO] Falha ao processar {name}: {count_or_err}")
        return

    if args.input_dir:
        image_extensions = (".jpg", ".jpeg", ".png", ".webp")
        all_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.lower().endswith(image_extensions):
                    full_in = os.path.join(root, f)
                    rel_path = os.path.relpath(full_in, args.input_dir)
                    all_files.append((full_in, rel_path))

        all_files.sort(key=lambda x: x[1])
        total = len(all_files)
        if total == 0:
            print(f"Nenhuma imagem encontrada na pasta: {args.input_dir}")
            return

        print(f"Iniciando processamento em lote de {total} imagens com {args.workers} workers paralelos...")
        start_time = time.time()

        tasks = []
        for in_path, rel_path in all_files:
            rel_dir = os.path.dirname(rel_path)
            fname = os.path.basename(rel_path)
            out_fname = f"comparacao_{fname}" if args.preview else fname
            out_path = os.path.join(args.output_dir, rel_dir, out_fname)
            tasks.append((in_path, out_path, args.preview))

        successes = 0
        plates_found = 0

        # Execute using ProcessPoolExecutor for maximum multi-core CPU speed
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, (ok, fname, val) in enumerate(executor.map(process_single_file, tasks), 1):
                if ok:
                    successes += 1
                    plates_found += (1 if val > 0 else 0)
                    pct = (idx / total) * 100
                    status = f"{val} placa(s) limpa(s)" if val > 0 else "sem placa (inalterada)"
                    print(f"[{idx:04d}/{total:04d}] ({pct:5.1f}%) {fname}: {status}")
                else:
                    print(f"[{idx:04d}/{total:04d}] ERRO em {fname}: {val}")

        total_time = time.time() - start_time
        print("\n==========================================")
        print(f"Processamento em lote concluído!")
        print(f"Total de imagens: {total}")
        print(f"Sucesso: {successes}/{total}")
        print(f"Imagens com placas tratadas: {plates_found}")
        print(f"Tempo total: {total_time:.2f} segundos")
        print(f"Média por imagem: {(total_time / total) * 1000:.1f} ms")
        print(f"Imagens salvas na pasta: {os.path.abspath(args.output_dir)}")
        print("==========================================")

if __name__ == "__main__":
    main()
