#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Слепое прослушивание 30 расхождений формы заполнителя (ыыы / ііі).

Для каждой записи: проигрывается аудио, показывается контекст, где [ ? ] —
тот самый заполнитель, и его номер среди слов. Вы решаете, какой ГЛАСНЫЙ
слышен в этой паузе: задний ы-тип или передний і-тип. Какой вариант
принадлежит транскрибатору, а какой модели — не показывается.

Клавиши:  a / b — вариант A или B
          n     — различить нельзя (нейтральный звук, [ə], шум)
          r     — повторить
          q     — выйти с сохранением
После ответа: уверенность 1 (высокая), 2 (средняя), 3 (низкая).

Запуск из корня корпуса:
    python3 code/listen_fillers.py filler_audit.csv
"""
import csv, os, re, subprocess, sys, termios, tty


def getkey():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd); ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def audio_path(record_id):
    m = re.search(r'(?:\.txt)+$', record_id)
    if not m:
        return record_id + '.flac'
    return record_id[:m.start()] + '.flac' * (len(m.group(0)) // 4)


def play(path):
    for cmd in (['afplay', path], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path]):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'filler_audit.csv'
    dst = os.path.splitext(src)[0] + '.judged.csv'
    rows = list(csv.DictReader(open(dst if os.path.exists(dst) else src, encoding='utf-8')))
    fields = list(rows[0].keys())

    def save():
        with open(dst, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    todo = [r for r in rows if not r['ОТВЕТ']]
    print('Всего {}, осталось {}. Клавиши: a / b / n(нельзя) / r(повтор) / q(выход)\n'.format(len(rows), len(todo)))
    for i, r in enumerate(todo, 1):
        path = audio_path(r['record_id'])
        while True:
            print('=' * 66)
            print('[{}/{}] {}   заполнитель — слово {}'.format(i, len(todo), r['id'], r['слово_номер']))
            print('  ... {} ...'.format(r['контекст']))
            print('  A = {}      B = {}'.format(r['вариант_A'], r['вариант_B']))
            if not play(path):
                print('!! аудио не найдено: ' + path)
            k = getkey()
            if k == 'r':
                continue
            if k == 'q':
                save(); print('\nСохранено в ' + dst); return
            if k in ('a', 'b', 'n'):
                r['ОТВЕТ'] = k.upper()
                print('  -> {}. Уверенность: 1 высокая / 2 средняя / 3 низкая'.format(r['ОТВЕТ']))
                c = ''
                while c not in ('1', '2', '3'):
                    c = getkey()
                r['УВЕРЕННОСТЬ'] = c
                save(); print()
                break
    save()
    print('\nГотово: ' + dst)


if __name__ == '__main__':
    main()
