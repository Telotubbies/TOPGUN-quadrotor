% function [X,Y,Z,Psi] = getTrajectory(t,trajectory_coef,timepoints)
% 
% tmp = [];
% for power = 0:7;
%     tmp = [tmp t.^power];
% end
% 
% 
% 
% i = 1;
% while 1
%     if t <= timepoints(i+1)
%      X = tmp*trajectory_coef(1:8,i);
%      Y = tmp*trajectory_coef(9:16,i);
%      Z = tmp*trajectory_coef(17:24,i);
%      Psi = tmp(1,1:4)*trajectory_coef(25:28,i);
%      break
% 
%     elseif t > timepoints(i+1)
%         i = i+1;
%     end
% 
% end

function [X,Y,Z,Psi,dX,dY,dZ,dPsi,ddX,ddY,ddZ,ddPsi,dddX,dddY,dddZ,ddddX,ddddY,ddddZ] = getTrajectory(t,trajectory_coef,timepoints)

tmp = [];
for power = 0:7;
    tmp = [t.^power tmp];
end



i = 1;
while 1
    if t <= timepoints(i+1)
     % X = tmp*flip(trajectory_coef(1:8,i),1);
     % Y = tmp*flip(trajectory_coef(9:16,i),1);
     % Z = tmp*flip(trajectory_coef(17:24,i),1);
     % Psi = tmp(1,1:4)*flip(trajectory_coef(25:28,i),1);
     % dX = tmp(1,2:8).*polyder(fliplr(trajectory_coef(1:8,i)'));
     % dY = tmp(1,2:8).*polyder(fliplr(trajectory_coef(9:16,i)'));
     % dZ = tmp(1,2:8).*polyder(fliplr(trajectory_coef(17:24,i)'));
     % dPsi = tmp(1,6:8).*polyder(fliplr(trajectory_coef(25:28,i)'));
     % ddX = tmp(1,3:8).*polyder(polyder(fliplr(trajectory_coef(1:8,i)')));
     % ddY = tmp(1,3:8).*polyder(polyder(fliplr(trajectory_coef(9:16,i)')));
     % ddZ = tmp(1,3:8).*polyder(polyder(fliplr(trajectory_coef(17:24,i)')));
     % ddPsi = tmp(1,7:8).*polyder(polyder(fliplr(trajectory_coef(25:28,i)')));
     tmpX = flip(trajectory_coef(1:8,i),1);
     % tmpX = tmpX';
     tmpY = flip(trajectory_coef(9:16,i),1);
     % tmpY = tmpY';
     tmpZ = flip(trajectory_coef(17:24,i),1);
     % tmpZ = tmpZ';
     tmpPsi = flip(trajectory_coef(25:28,i),1);
     % tmpPsi = tmpPsi';
     X = tmp*tmpX;
     Y = tmp*tmpY;
     Z = tmp*tmpZ;
     Psi = tmp(1,5:8)*tmpPsi;
     tmpX = polyder(tmpX');
     tmpY = polyder(tmpY');
     tmpZ = polyder(tmpZ');
     tmpPsi = polyder(tmpPsi');
  
     dX = tmp(1,2:8)*tmpX';
     dY = tmp(1,2:8)*tmpY';
     dZ = tmp(1,2:8)*tmpZ';
     dPsi = tmp(1,6:8)*tmpPsi';

     tmpX = polyder(tmpX);
     tmpY = polyder(tmpY);
     tmpZ = polyder(tmpZ);
     tmpPsi = polyder(tmpPsi);

     ddX = tmp(1,3:8)*tmpX';
     ddY = tmp(1,3:8)*tmpY';
     ddZ = tmp(1,3:8)*tmpZ';
     ddPsi = tmp(1,7:8)*tmpPsi';

     tmpX = polyder(tmpX);
     tmpY = polyder(tmpY);
     tmpZ = polyder(tmpZ);
     

     dddX = tmp(1,4:8)*tmpX';
     dddY = tmp(1,4:8)*tmpY';
     dddZ = tmp(1,4:8)*tmpZ';

     tmpX = polyder(tmpX);
     tmpY = polyder(tmpY);
     tmpZ = polyder(tmpZ);
     

     ddddX = tmp(1,5:8)*tmpX';
     ddddY = tmp(1,5:8)*tmpY';
     ddddZ = tmp(1,5:8)*tmpZ';
   


     break

    elseif t > timepoints(i+1)
        i = i+1;
    end

end





end