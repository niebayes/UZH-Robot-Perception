function imagerot = imrotate( image, degree )

  if(degree~=0)
      rads=degree*pi/180;
  
      rotate = maketform('affine',[ cos(rads)  sin(rads)  0; ...
                  -sin(rads)  cos(rads)  0; ...
                  0       0       1 ]);
  
      so = size(image);
      twod_size = so(1:2);
  
      [loA,hiA,loB,hiB,outputSize] = getOutputBound(rotate,twod_size);
  
      %image padding
      [Rows, Cols] = size(image); 
      % Diagonal = sqrt(Rows^2 + Cols^2); 
      % RowPad = ceil(Diagonal - Rows) + 2;
      % ColPad = ceil(Diagonal - Cols) + 2;
      offRow=0;
      RowPad=outputSize(1)-Rows;
      if(RowPad<=0)
          offRow=1+abs(RowPad)/2;
          RowPad=1;        
      end
  
      ColPad=outputSize(2)-Cols;
  
      offCol=0;
      if(ColPad<=0)
          offCol=1+abs(ColPad)/2;
          ColPad=1;        
      end
      
      imagepad = zeros(Rows+RowPad, Cols+ColPad);
      imagepad(ceil(RowPad/2):(ceil(RowPad/2)+Rows-1),ceil(ColPad/2):(ceil(ColPad/2)+Cols-1)) = image;
  
      %midpoints
      midx=ceil((size(imagepad,1)+1)/2);
      midy=ceil((size(imagepad,2)+1)/2);
  
      imagerot=zeros(outputSize); % midx and midy same for both
  
      for i=1:size(imagerot,1)
          for j=1:size(imagerot,2)
  
              posi=i-midx+offRow;
              posj=j-midy+offCol;
               x= (posi)*cos(rads)+(posj)*sin(rads);
               y=-(posi)*sin(rads)+(posj)*cos(rads);
               x=round(x)+midx-1;
               y=round(y)+midy-1;
  
               if (x>=1 && y>=1 && x<=size(imagepad,1) && y<=size(imagepad,2))
                    imagerot(i,j)=imagepad(x,y); % k degrees rotated image         
               end
  
          end
      end
  
      imagerot=uint8(imagerot);
  
  else
     imagerot=image; 
  end
  
  end
  
  
  
  function [loA,hiA,loB,hiB,outputSize] = getOutputBound(rotate,twod_size)
  
  % Coordinates from center of A
  hiA = (twod_size-1)/2;
  loA = -hiA;
  hiB = ceil(max(abs(tformfwd([loA(1) hiA(2); hiA(1) hiA(2)],rotate)))/2)*2;
  loB = -hiB;
  outputSize = hiB - loB + 1;
  
  end