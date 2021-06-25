from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 오목판, 돌 이미지 가져오기
path = 'C:\\Users\\thsdb\\AppData\\Local\\Programs\\Python\\Python39\\MyProject\\오목AI\\'

board_img = np.array(Image.open(path+'오목판.png')) # (594, 608, 4)
b_ball = np.array(Image.open(path+'흑돌.png'))# 일단은 흑돌로 -> 백돌도 따로 구분해서 하는 코드 추가해야함
w_ball = np.array(Image.open(path+'백돌.png'))

class Omok_Simoulator:
    # 이미지를 그린후 반영
    def game_visualize(slef, board_img, ball, pos_H, pos_W): # 호출되면 해당 position에 돌 놓고 print

        ball_size = ball.shape[0] # (70, 70, 4)
        step_size = 48 # 49이고
        off_set = 7 # 8방향인데?

        y_step = step_size * pos_H - round(step_size/2) + off_set
        x_step = step_size * pos_W - round(step_size/2) + off_set
        
        board_img[y_step:y_step+ball_size, x_step:x_step+ball_size] = ball

        plt.imshow(board_img)
        plt.axis('off')
        plt.show()

    # 오목의 승리자를 결정
    def game_rule(self, board, player): # 다섯줄을 먼저 만들면 승리 (다른 규칙 X)
        
        game_result = 0
        diag_line = np.zeros(5)

        # 가로 5줄 검출 코드
        for i_idx in range(len(board)):
            for j_idx in range(len(board)-4):
                p1 = (board[i_idx, j_idx:j_idx+5] == player)

                if p1.sum() == 5:
                    game_result = 1
                    
                    return game_result
        
        # 세로 5줄 검출 코드
        for i_idx in range(len(board)-4):
            for j_idx in range(len(board)):
                p1 = (board[i_idx:i_idx+5, j_idx] == player)

                if p1.sum() == 5:
                    game_result = 1

                    return game_result
        
        # 대각선 5줄 검출 코드
        for i_idx in range(len(board)-4):
            for j_idx in range(len(board)-4):

                diag_line[0] = board[i_idx+0, j_idx+0]
                diag_line[1] = board[i_idx+1, j_idx+1]
                diag_line[2] = board[i_idx+2, j_idx+2]
                diag_line[3] = board[i_idx+3, j_idx+3]
                diag_line[4] = board[i_idx+4, j_idx+4]

                p1 = (diag_line == player)

                if p1.sum() == 5:
                    game_result = 1

                    return game_result

# 게임 판 크기 7x7
size_of_board = 7
board_array = np.zeros([size_of_board, size_of_board])

max_turn = size_of_board * size_of_board # 최대 라운드?

Omok_S = Omok_Simoulator()

for i in range(4): # 0 ~ 48

    if i%2 ==0: # 흑돌이 돌을 둠
        pos_H, pos_W = map(int, input('흑돌 차례입니다.').split(' '))

        board_array[pos_H, pos_W] = 1 # 1은 흑돌

        Omok_S.game_visualize(board_img, b_ball, pos_H, pos_W)
    
    else:
        pos_H, pos_W = map(int, input('백돌 차례입니다.').split(' '))

        board_array[pos_H, pos_W] = 2 # 2는 백돌

        Omok_S.game_visualize(board_img, w_ball, pos_H, pos_W)
    
    print(board_array)
    # 흑돌이 백돌을 덮어버리는 등 오류는 추가 수정