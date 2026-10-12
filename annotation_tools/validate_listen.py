#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прослушивание для формальной валидации и аудиопроверки.

Валидация (200 записей): решаете, спонтанная речь или подготовленная.
    python3 validate_listen.py validation_B.csv
        s — спонтанная      p — подготовленная / читаемая
        n — нет сигнала (слишком коротко, не понять)
        x — неразборчиво, аудио непригодно
        затем уверенность: 1 высокая, 2 средняя, 3 низкая

Аудиопроверка (100 записей): есть ли в ЗВУКЕ заминка, которой нет в ТЕКСТЕ.
    python3 validate_listen.py spotcheck_100.csv
        y — да, слышна заминка/повтор/обрыв, которых нет в тексте
        n — нет, текст передаёт речь без пропусков

Общие клавиши: r — повторить, q — выйти (прогресс сохраняется).
Работает на macOS (afplay) и Linux (ffplay/aplay). Результат: <файл>.judged.csv
"""
import csv, os, re, subprocess, sys, termios, tty


def key():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd); c = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return c.lower()


def audio(rid):
    m = re.search(r'(?:\.txt)+$', rid)
    return rid + '.flac' if not m else rid[:m.start()] + '.flac' * (len(m.group(0)) // 4)


def play(p):
    for cmd in (['afplay', p], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', p], ['aplay', '-q', p]):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main():
    src = sys.argv[1]
    dst = os.path.splitext(src)[0] + '.judged.csv'
    rows = list(csv.DictReader(open(dst if os.path.exists(dst) else src, encoding='utf-8')))
    fields = list(rows[0].keys())
    spot = 'ОТВЕТ' in fields
    col = 'ОТВЕТ' if spot else 'ОЦЕНКА'
    valid = {'y': 'Д', 'n': 'Н'} if spot else {'s': 'S', 'p': 'P', 'n': 'NS', 'x': 'X'}

    def save():
        with open(dst, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    todo = [r for r in rows if not r[col]]
    print('Всего {}, осталось {}.'.format(len(rows), len(todo)))
    print('Клавиши: ' + ('y да / n нет' if spot else 's спонт. / p подгот. / n нет сигнала / x неразборчиво')
          + ' / r повтор / q выход\n')
    for i, r in enumerate(todo, 1):
        while True:
            print('=' * 66)
            print('[{}/{}] {}'.format(i, len(todo), r['id']))
            print('  ' + (r['text'] if spot else r['preview'])[:240])
            if not play(audio(r['record_id'])):
                print('!! нет аудио: ' + audio(r['record_id']))
            k = key()
            if k == 'r':
                continue
            if k == 'q':
                save(); print('\nСохранено: ' + dst); return
            if k in valid:
                r[col] = valid[k]
                if not spot and k in ('s', 'p'):
                    print('  -> {}. Уверенность: 1 / 2 / 3'.format(r[col]))
                    c = ''
                    while c not in ('1', '2', '3'):
                        c = key()
                    r['УВЕРЕННОСТЬ'] = c
                save(); print('  -> ' + r[col] + '\n')
                break
    save(); print('\nГотово: ' + dst)


if __name__ == '__main__':
    main()
